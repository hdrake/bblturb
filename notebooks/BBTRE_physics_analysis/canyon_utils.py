from xgcm import Grid
import xarray as xr
import numpy as np

def add_thalweg(ds):
    ds['YC_thalweg'] = xr.DataArray(
        np.array([
            ds['YC'].isel(
                YC=(ds['Depth'].isel(XC=i) == ds['Depth'].isel(XC=i).max(dim='YC'))
            ).values[0]
            for i in range(ds['XC'].size)
        ]),
        coords=(ds['XC'],)
    )
    
    ds['j_thalweg'] = xr.DataArray(
        np.array([
            np.where(ds['Depth'].isel(XC=i) == ds['Depth'].isel(XC=i).max(dim='YC'))[0][0]
            for i in range(ds['XC'].size)
        ]),
        coords=(ds['XC'],)
    )
    return ds