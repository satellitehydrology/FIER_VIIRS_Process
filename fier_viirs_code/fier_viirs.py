import numpy as np

import xarray as xr
import rioxarray
import rasterio

import pandas as pd

import geopandas as gpd
from shapely.geometry import mapping
from operator import itemgetter

import matplotlib.pyplot as plt

import os
import glob


def pile_org_viirs_wf(org_viirs_wf_folder: str, aoi_tile_list: list, aoi_shape: gpd.geodataframe.GeoDataFrame, start_date: str, end_date: str) -> xr.Dataset:

    """
    Function to pile up original VIIRS water fraction maps.

    args:
        org_viirs_wf_folder: Folder with original VIIRS water fraction 
        (downloaded from https://noaa-jpss.s3.amazonaws.com/index.html#JPSS_Blended_Products/VFM_1day_GLB/NetCDF/)

        aoi_tile_list: Tile list of AOI (As a list because AOI may cover more than 1 tile. Users will have to visually check that)
        aoi_shape: GeoJson format of AOI
        start_date: String (YYYY-mm-dd) of start date for piling up
        end_date: String (YYYY-mm-dd) of end date for piling up

    output:
        org_img_stack: Piled-up VIIRS water fraction maps
    """

    date_range = pd.date_range(start=start_date, end=end_date)

    bounds = aoi_shape.geometry.apply(lambda x: x.bounds).tolist()
    aoi_w, aoi_s, aoi_e, aoi_n = min(bounds, key=itemgetter(0))[0], min(bounds, key=itemgetter(1))[1], max(bounds, key=itemgetter(2))[2], max(bounds, key=itemgetter(3))[3]

    ct_valid_img = 0        
    for pd_datetime in date_range:

        # ----- Get the tile(s) that encompasses AOI. Tiles will be mosaicked if needed -----
        ct_tile = 0    
        for tile in aoi_tile_list:        
        
            data_dir = org_viirs_wf_folder + '\\*-GLB' + str(tile).zfill(3) + '_v1r0_blend_s'+pd_datetime.strftime('%Y%m%d')+'*.nc'
                
            # ----- Get file list in the directory -----
            flist = glob.glob(data_dir)
            try: 
                img_temp = xr.open_dataset(flist[0]).WaterDetection           
            except:
                break
            
            ct_tile = ct_tile+1
            if tile==aoi_tile_list[0]:
                img = img_temp
            else:  
                img = xr.combine_by_coords([img, img_temp], combine_attrs='drop') 
                
        if ct_tile!=len(aoi_tile_list):
            continue           
    
        #pd_datetime = pd.DatetimeIndex([pd.to_datetime(date_str).strftime('%Y-%m-%d')])


        ### Need to check how to get aoi_w, aoi_e, aoi_s, aoi_n
        
        c_w = np.argwhere(img.lon.values>=aoi_w)[0][0]
        c_e = np.argwhere(img.lon.values<=aoi_e)[-1][0]
        r_s = np.argwhere(img.lat.values>=aoi_s)[-1][0]
        r_n = np.argwhere(img.lat.values<=aoi_n)[0][0]
        
        try:
            img = img.WaterDetection
        except:
            img = img
    
        img = img[r_n:r_s,c_w:c_e]
        img.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        img.rio.write_crs("epsg:4326", inplace=True)
        aoi_img = img.rio.clip(aoi_shape.geometry.apply(mapping), aoi_shape.crs, drop=True)        
        del img
    
        ct_valid_img = ct_valid_img+1
        
        aoi_img = aoi_img.expand_dims('time', axis=0)                 
        if ct_valid_img==1:  
            aoi_stack = aoi_img            
            time = pd.DatetimeIndex([pd_datetime.strftime('%Y%m%d')])
        else:                     
            aoi_stack = xr.concat([aoi_stack,aoi_img], dim='time')
            time = time.append(pd.DatetimeIndex([pd_datetime.strftime('%Y%m%d')]))

    
    org_img_stack = xr.Dataset(
        data_vars=dict(
            water_fraction=(["time","lat", "lon"], aoi_stack.values),
        ),
        coords=dict(
            time=time, #.strftime('%Y-%m-%d'),
            lon=(["lon"], aoi_stack.lon.values),
            lat=(["lat"], aoi_stack.lat.values)                
        ),
    )

    return org_img_stack








def prep_viirs_wf(original_wf: xr.Dataset, for_reof_opt: bool, non_cloud_per: float=0.95) -> xr.DataArray: #(original_wf, max_fld_indx, for_reof_opt, non_cloud_per=0.95):
    
    """
    Function to preprocess piled-up JPSS water fraction maps for either REOF analysis or being used as reference of quantile-scaling/validation

    args:
        original_wf: Original JPSS water fraction in XARRAY Dataset format (output of "pile_org_viirs_wf")
        max_fld_indx: Historical maximum flooded extents
        for_reof_opt: If the output will be used for REOF analysis
        non_cloud_per: Ratio of non-cloud pixels in the historical maximum flooded extents

    output:
        mdf_wf_da: pre-edited VIIRS water fraction maps for either REOF analysis or being used as reference of quantile-scaling/validation
    
    """
    
    from rasterio.fill import fillnodata


    def prepare_wf(org_full_stack, valid_per_thrs, for_reof_opt): #(org_full_stack, max_fld_indx, valid_per_thrs, for_reof_opt):            

        # Pre-edit pixel values of NOAA VIIRS water fraction map
        def pre_edit_wf(in_img_stack, step, for_reof_opt):
            
            if step == 1: 
                # For counting the cloud-relevant pixels, and making them as NaN
                img_stack = np.where(in_img_stack==1, np.nan, in_img_stack) # No-data        
                img_stack = np.where(img_stack==30, np.nan, img_stack) # Cloud
                img_stack = np.where(img_stack==50, np.nan, img_stack) # Cloud shadow
                    
            elif step == 2:
                # Edit the pixels that are not cloud-relevant but have no water fraction values
                img_stack = np.where(in_img_stack==16, 0, in_img_stack) # Bareland
                img_stack = np.where(img_stack==17, 0, img_stack) # Vegetation
        
                if for_reof_opt==True: 
                    # Generate the image cube used for REOF analysis 
                    # (If we make all of these pixels as NaN, many images in Spring will be removed)
                    img_stack = np.where(img_stack==20, 0, img_stack) # Snow
                    img_stack = np.where(img_stack==25, 0, img_stack) # Ice
                    img_stack = np.where(img_stack==27, 0, img_stack) # River and lake ice
                    img_stack = np.where(img_stack==38, 0, img_stack) # Supra-snow/ice water, mixed water, melting ice
                    img_stack = np.where(img_stack==15, 0, img_stack) # Floodwater without water fraction                
            
                else: 
                    # Generate the image cube used as the reference for the quantile-mapping process and validation 
                    # (Only retain the pixels that have real water fraction values)
                    img_stack = np.where(img_stack==20, np.nan, img_stack)
                    img_stack = np.where(img_stack==25, np.nan, img_stack)   
                    img_stack = np.where(img_stack==27, np.nan, img_stack)
                    img_stack = np.where(img_stack==38, np.nan, img_stack)
                    img_stack = np.where(img_stack==15, np.nan, img_stack) 
                   
        
                # The original water fractions are labeled as 100 to 200, representing 0 to 100 %. 
                # Thus, the original values are subtracted by 100.
                img_stack = np.where(img_stack==99, 200, img_stack) # Make normal open water as 200. It will be subtracted by 100   
                img_stack = np.where(img_stack>=100, img_stack-100, img_stack)   
        
            return img_stack
    
        
        # ----- Get shapes -----
        stack_shape = org_full_stack.values.shape
        spatial_shape = stack_shape[1:]
        flat_shape = (stack_shape[0],np.prod(spatial_shape))
    
        # ----- Flatten the data from [t,y,x] to [t, y*x] and remove pixels that are always NaN -----
        org_full_flat = org_full_stack.values.reshape(flat_shape)                
        aoi_mask_flat = ~(np.isnan(org_full_flat).all(axis=0))        
        org_full_flat = org_full_flat[:, aoi_mask_flat]
        aoi_mask = aoi_mask_flat.reshape(spatial_shape)
        flat2geo = np.arange(np.prod(spatial_shape))[aoi_mask_flat].reshape(1,-1)    
        
        # ----- Pre-editing 1 -----
        # -- Make cloud-relevant pixels as NaN to easily count their number and keep only less cloudy images --
        mdf_img_flat0 = pre_edit_wf(org_full_flat, 1, for_reof_opt)   
        if for_reof_opt:
            # -- Retain less cloudy images ( <5 % cloud-relevant pixels in the AOI) --
            temp_mdf_img_flat0 = (mdf_img_flat0.copy())[:,aoi_mask_flat[aoi_mask_flat==1]]
            valid_pix_perc = (1-(np.count_nonzero(np.isnan(temp_mdf_img_flat0), axis=1)/temp_mdf_img_flat0.shape[1]))
            time_mask_flat = valid_pix_perc >= valid_per_thrs
            mdf_img_flat = mdf_img_flat0[time_mask_flat,:]
            flat2time = np.arange(np.prod(stack_shape[0]))[time_mask_flat].reshape(-1,1)
            img_datetime = org_full_stack.time.values[time_mask_flat]         
        else:
            mdf_img_flat = mdf_img_flat0   
            img_datetime = org_full_stack.time.values  
    
            
        # ----- Pre-editing 2 -----
        # -- Edit other pixels that do not have water fraction values (pixels "labeled" as snow, ice, and so on) --
        img_flat = pre_edit_wf(mdf_img_flat, 2, for_reof_opt) 
    
        
        # ----- Reshape the flattened array back to the geographical dimension -----
        fill_img_flat = np.ones((img_flat.shape[0], flat_shape[1]), dtype=np.float32)*np.nan
        for ct_c in np.arange((img_flat.shape[1])):
            fill_img_flat[:,flat2geo[0,ct_c]] = img_flat[:,ct_c] #change this to the DINEOFed array
        rec_mdf_img = fill_img_flat.reshape(img_flat.shape[0], stack_shape[1], stack_shape[2])
        
    
        # create output dataset
        mdf_wf_da = xr.DataArray(
            data=rec_mdf_img, 
            coords={'time': (['time'], img_datetime),
                    'lat': (['lat'], org_full_stack.lat.values,{'units':'degree'}),
                    'lon': (['lon'], org_full_stack.lon.values,{'units':'degree'})                 
                   }, 
            attrs={'long name':'water fraction', 
                   'unit':'%',
                   '_FillValue':np.nan}
        )
        mdf_wf_da = mdf_wf_da.rio.write_crs(4326)
    
        
        return mdf_wf_da, aoi_mask


    
    if for_reof_opt==True:
        print('Prepare data for REOF analysis')
        # ----- Pre-editing -----
        mdf_wf_da, aoi_mask = prepare_wf(original_wf.water_fraction, non_cloud_per, for_reof_opt) #(org_full_stack, max_fld_indx, non_cloud_per, for_reof_opt)

        # ----- Spatial interpolation -----
        for ct_time in np.arange(mdf_wf_da.sizes['time']):
            nan_in_aoi = ~np.logical_and(np.isnan(mdf_wf_da[ct_time]), aoi_mask)
            mdf_wf_da[ct_time] = fillnodata(mdf_wf_da[ct_time], nan_in_aoi)
        
        mdf_wf_da = mdf_wf_da.where(mdf_wf_da>0,0)
        mdf_wf_da = mdf_wf_da.where(mdf_wf_da<100,100)
        mdf_wf_da = mdf_wf_da.where(aoi_mask)                 
    else:
        print('Prepare data as reference of quantile-scaling or validation')
        # ----- Pre-editing -----
        mdf_wf_da, aoi_mask = prepare_wf(original_wf.water_fraction, 0, for_reof_opt)  #(org_full_stack, max_fld_indx, 0, for_reof_opt) 

    return mdf_wf_da




def perf_qm_mon_ext(hist_obs_stack: xr.DataArray, hist_syn_stack: xr.DataArray, fct_syn_stack: xr.DataArray, nbins: int=100) -> xr.DataArray:

    """
    Function for quantile-scaling

    args:
        hist_obs_stack: Observed historical VIIRS water fraction stack used as reference
        hist_syn_stack: FIER-synthesized historical VIIRS water fraction
        fct_syn_stack:  FIER-synthesized forecasted VIIRS water fraction
        nbins: Number of bins for CDF

    output:
        map_fct_syn: quantile-scaled FIER-synthesized forecasted VIIRS water fraction
    """

    hist_obs_stack = hist_obs_stack.where(hist_obs_stack.time.isin(hist_syn_stack.time),drop=True)
    hist_syn_stack = hist_syn_stack.where(hist_syn_stack.time.isin(hist_obs_stack.time),drop=True)    
    
    map_fct_syn = np.empty((fct_syn_stack.sizes['time'],fct_syn_stack.sizes['lat'],fct_syn_stack.sizes['lon']))
    map_fct_syn[:] = np.nan    
    
    binmid = np.arange(0, 1.+1./nbins, 1./nbins)
    
    obs = hist_obs_stack.values 
    syn = np.where(np.isnan(obs), np.nan, hist_syn_stack.values)
    
    size_uniq_hist_yr = np.unique(pd.to_datetime(hist_obs_stack.time.values).year).size
    hist_mon = pd.to_datetime(hist_obs_stack.time.values).month
    
    fct_syn_stack_time = fct_syn_stack.time
     
    fct_mon = pd.to_datetime(fct_syn_stack.time.values).month    
    uniq_fct_mon = np.unique(fct_mon)    
    
    for mon in uniq_fct_mon:
        
        prev_mon = mon-1
        next_mon = mon+1
        
        if mon==1:            
            indx_hist_mon = np.logical_or(np.logical_and(hist_mon>=mon, hist_mon<=next_mon), hist_mon==12)
        elif mon==12:
            indx_hist_mon = np.logical_or(np.logical_and(hist_mon>=prev_mon, hist_mon<=mon), hist_mon==1)
        else:
            indx_hist_mon = np.logical_and(hist_mon>=prev_mon, hist_mon<=next_mon)
        indx_fct_mon = fct_mon==mon
        
        obs_mon = obs[indx_hist_mon, :, :]
        syn_mon = syn[indx_hist_mon, :, :]  
        qm_mon = fct_syn_stack.values[indx_fct_mon,:,:]        
                    
        qobs = np.nanquantile(obs_mon, binmid, axis=0)
        qsyn = np.nanquantile(syn_mon, binmid, axis=0)         
    
        for ct_r in range(hist_obs_stack.sizes['lat']):
            for ct_c in range(hist_obs_stack.sizes['lon']):

                bin_unc_mdl = np.interp(qm_mon[:, ct_r, ct_c], qsyn[:,ct_r,ct_c], binmid)                 
                map_fct_syn[indx_fct_mon,ct_r,ct_c] = np.interp(bin_unc_mdl, binmid, qobs[:, ct_r, ct_c])                    
                    

    map_fct_syn[map_fct_syn>100] = 100
    map_fct_syn[map_fct_syn<0] = 0
    
    map_fct_syn = xr.DataArray(

        data=map_fct_syn,
        coords=dict(
            time=(["time"], fct_syn_stack.time.values),
            lat=(["lat"], hist_obs_stack.lat.values), 
            lon=(["lon"], hist_obs_stack.lon.values)
                           
        ),

    )    
        
    return map_fct_syn
