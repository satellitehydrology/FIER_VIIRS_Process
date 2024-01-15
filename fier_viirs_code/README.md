# Overview

**fier_viirs.py** includes the functions of

1. Read original VIIRS water fraction maps in the given time span, clip them to the given AOI, and pile them up as a multi-temporal image cube. Mosaicking are performed if the AOI covers more than 1 VIIRS water fraction tiles.
2. Pre-edit original VIIRS water fraction image cube for the purpose of REOF analysis, or being used as reference to quantile scale the FIER-synthesized water fraction maps.
3. Quantile-scale the FIER-synthesized water fraction maps.

**Note that functions relevant to the FIER process itself (REOF analysis and synthesis) are in the FIERpy repo/package.**
