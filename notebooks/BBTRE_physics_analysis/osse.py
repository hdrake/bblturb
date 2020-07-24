from xgcm import Grid
import xarray as xr
import numpy as np

def irregular_section(
        da, section_idx,
        section_dim="XC", sample_dim="YC",
        width=0, return_mean=False, invert_mask=False
    ):
    stacked = da.stack(XY=(section_dim, sample_dim))
    stacked_idx = xr.DataArray(np.zeros(stacked.XY.shape, dtype=bool), coords={'XY':stacked.coords['XY']}, dims='XY')
    for i, j in enumerate(section_idx):
        ii = i*da[sample_dim].size
        j_min = (j-width)%(da[sample_dim].size)
        j_max = (j+width)%(da[sample_dim].size)
        if j_min+ii < j_max+ii+1:
            stacked_idx[j_min+ii: j_max+ii+1] = True
        else:
            stacked_idx[ii:j_max+ii+1] = True
            stacked_idx[j_min+ii:ii+da[sample_dim].size] = True
    if invert_mask:
        tmp=stacked.where(~stacked_idx).unstack()
    else:
        tmp=stacked.where(stacked_idx).unstack()
    if (width==0) | return_mean:
        return tmp.mean(dim=sample_dim, skipna=True)
    else:
        return tmp
    
    
def global_argmin(da):
    xmin = da.min(dim=['Xr'])
    xargmin = da.argmin(dim='Xr').values
    yargmin = xmin.argmin(dim='Yr').values
    return [np.int(xargmin[yargmin]), np.int(yargmin)], xmin[yargmin].values

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def out_of_bounds(domain, sample_indices):
    return (
        (sample_indices[0] == 0) | (sample_indices[0] == domain.dims['Xr']-1) |
        (sample_indices[1] == 0) | (sample_indices[1] == domain.dims['Yr']-1)
    )

def isnumber(item):
    return isinstance(item, np.int) | isinstance(item, np.float)

def sample_locations(domain, lons, lats):
    samples = {"i":[], "j":[], "lon":[], "lat":[]}
    for lon, lat in zip(lons, lats):
        if not(isnumber(lon) & isnumber(lat)): continue
        dist = distance(domain['lon'], domain['lat'], lon, lat)
        argmin, _ = global_argmin(dist)
        if out_of_bounds(domain, argmin): continue
        samples["i"].append(argmin[0])
        samples["j"].append(argmin[1])
        samples["lon"].append(lon)
        samples["lat"].append(lat)
    return samples
        