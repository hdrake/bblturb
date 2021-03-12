from xgcm import Grid
import xarray as xr
import numpy as np

import matplotlib
import copy
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 16

# Constants
g = 9.81 # gravitational acceleration
ρ0 = 1000. # reference density
α = 2.e-4 # thermal expansion coefficient
day2seconds = 1./86400.

# Plotting parameters
nancol = (0.65,0.65,0.65)
div_cmap = copy.copy(matplotlib.cm.get_cmap("RdBu_r"))

div_cmap.set_bad(color=nancol)
cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap.set_bad(color=nancol)

def mean_profile(da):
    da_m = da.where(da != 0.)
    return da_m.mean(dim=['XC', 'YC'], skipna=True)

def weighted_mean(da, weights, **kwargs):
    return (da*weights).sum(**kwargs)/weights.sum(**kwargs)

def add_background_temp(ds, Γ):
    ds['THETA'] = ds['THETA'].where(ds['THETA'] != 0.)
    ds['THETA_BG'] = ds['THETA'] + ds['Zr'] * Γ
    return ds

def add_Nsq(ds, Γ):
    ds['Nsq'] = -g/ρ0*ds['DRHODR']
    ds['Nsq'] = ds['Nsq'].where(ds['Nsq'] != 0.)
    ds['Nsq'] += g*α*Γ
    return ds
    
def add_hab_coordinates(ds, grid, vert_coord='Z', gridface='C'):
    faceinterp = {'C':[], 'W':'X', 'S':'Y'}
    ds = ds.assign_coords({
        f'{vert_coord}_hab{gridface}':
        ds[vert_coord] + (grid.interp(ds['Depth'], faceinterp[gridface]) - ds['Depth'].max())
    })
    return ds

def _interp(x, y, bins=None):
    return np.interp(
        bins,
        x[::-1],
        y[::-1]
    )

def hab_interp(da, hab_bins=np.arange(2.5, 2000., 5.), vert_coord='Z', gridface='C'):
    hab = xr.DataArray(
        hab_bins,
        coords={f'hab{gridface}': hab_bins},
        dims=[f'hab{gridface}']
    )
    da_itp = xr.apply_ufunc(
        _interp,
        da[f'{vert_coord}_hab{gridface}'],
        da,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[vert_coord], [vert_coord]],
        output_core_dims=[[f'hab{gridface}']],
        output_sizes={f'hab{gridface}': hab.size},
        output_dtypes=[float],
        kwargs={'bins': hab_bins},
    )
    da_itp = da_itp.assign_coords({f'hab{gridface}': hab})
    return da_itp

def parallel_combine(ds_list, concat_dims):
    ds_new_list = []
    for concat_dim in concat_dims:
        tmp_list = []
        for ds in ds_list:
            tmp = ds.copy()
            for var in (list(ds.data_vars) + list(ds.coords)):
                if (concat_dim not in ds[var].dims) & any([(dim in concat_dims) for dim in ds[var].dims]):
                    tmp = tmp.drop_vars(var)
            tmp_list.append(tmp)
        ds_new_list.append(xr.combine_by_coords(tmp_list))
    return xr.merge(ds_new_list)

def periodic_extend(ds, concat_dims, L, extend_multiples):

    ds_list = []
    for extend in range(extend_multiples[0], extend_multiples[1]+1):
        tmp = ds.copy()
        tmp_attrs = [tmp[dim].attrs for dim in concat_dims]
        for dim in concat_dims:
            tmp[dim] = tmp[dim] + extend*L
        ds_list.append(tmp)
    ds = parallel_combine(ds_list, concat_dims = concat_dims)
    for i, dim in enumerate(concat_dims):
        ds[dim].attrs = tmp_attrs[i]
    return ds

def shift_vertical_coordinate(ds):
    H = ds['Depth'].max().values.copy().astype('>f8')
    ds.attrs["H"] = H
    for coord in ds.coords:
        if 'Z' in coord:
            tmp_attrs = ds[coord].attrs
            ds = ds.assign_coords({coord: ds[coord] + H})
            ds[coord].attrs = tmp_attrs
    return ds

def add_rotated_coords(ds, θ, shift_vertical = True):
    if shift_vertical:
        ds = shift_vertical_coordinate(ds)

    grid = Grid(ds, periodic=['X', 'Y'])
        
    ds = ds.assign_coords({
        'Zr': (['XC', 'Z'], (ds['XC']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
        'Xr': (['XC', 'Z'], (ds['XC']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
        'Zr_V': (['XC', 'Z'], (ds['XC']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
        'Xr_V': (['XC', 'Z'], (ds['XC']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
        'Zr_U': (['XG', 'Z'], (ds['XG']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
        'Xr_U': (['XG', 'Z'], (ds['XG']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
        'Zr_W': (['XC', 'Zl'], (ds['XC']*np.sin(θ) + ds['Zl']*np.cos(θ)).values),
        'Xr_W': (['XC', 'Zl'], (ds['XC']*np.cos(θ) - ds['Zl']*np.sin(θ)).values)
    })

    ds['Depthr'] = ds['Depth'] - ds['XC']*np.tan(θ)
    
    return ds, grid