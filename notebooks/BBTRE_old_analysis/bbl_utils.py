from xgcm import Grid
import xarray as xr
import numpy as np

import matplotlib
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
div_cmap = plt.get_cmap('RdBu_r')
div_cmap.set_bad(color=nancol)
cmap = plt.get_cmap('viridis')
cmap.set_bad(color=nancol)

def mean_profile(da):
    da_m = da.where(da != 0.)
    return da_m.mean(dim=['XC', 'YC'], skipna=True)

def weighted_mean(da, weights, **kwargs):
    return (da*weights).sum(**kwargs)/weights.sum(**kwargs)

# def pre_process(ds):
#     ds['time'] = (ds['time']*1.e-9).astype("float64")
#     ds['time'].attrs = {}
#     ds['time'].attrs['units'] = "s"
    
#     coords = ds.coords.to_dataset().reset_coords()
#     ds = ds.reset_coords(drop=True)
    
# #     # Add grid metrics
# #     coords['drW'] = coords.hFacW * coords.drF #vertical cell size at u point
# #     coords['drS'] = coords.hFacS * coords.drF #vertical cell size at v point
# #     coords['drC'] = coords.hFacC * coords.drF #vertical cell size at tracer point
# #     metrics = {
# #         ('X',): ['dxC', 'dxG'], # X distances
# #         ('Y',): ['dyC', 'dyG'], # Y distances
# #         ('Z',): ['drW', 'drS'], # Z distances
# #         ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
# #     }
# #     grid = Grid(coords, metrics=metrics, periodic=['X','Y'])
# #     # Add interpolated grid metrics
# #     coords['dxF'] = grid.interp(coords.dxC,'X')
# #     coords['dyF'] = grid.interp(coords.dyC,'Y')
# #     metrics = {
# #         ('X',): ['dxC', 'dxG', 'dxF'], # X distances
# #         ('Y',): ['dyC', 'dyG', 'dyF'], # Y distances
# #         ('Z',): ['drW', 'drS', 'drC'], # Z distances
# #         ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
# #     }
# #     grid = Grid(coords, metrics=metrics, periodic=['X','Y'])
    
#     grid = Grid(coords, periodic=['X','Y'])
    
#     H = coords['Depth'].max().values.copy().astype('>f8')
#     ds = ds.assign_coords(
#         Z = ds['Z'] + H,
#         Zl = ds['Zl'] + H,
#         Zu = ds['Zu'] + H,
#         Zp1 = ds['Zp1'] + H
#     )
    
#     coords = coords.assign_coords(
#         Z = coords['Z'] + H,
#         Zl = coords['Zl'] + H,
#         Zu = coords['Zu'] + H,
#         Zp1 = coords['Zp1'] + H
#     )

#     return ds, coords, grid


# def add_cartesian_coordinates(ds, coords, θ):
#     ds = ds.assign_coords({
#         'Zr': (['XC', 'Z'], (ds['XC']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
#         'Xr': (['XC', 'Z'], (ds['XC']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
#         'Zr_V': (['XC', 'Z'], (ds['XC']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
#         'Xr_V': (['XC', 'Z'], (ds['XC']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
#         'Zr_U': (['XG', 'Z'], (ds['XG']*np.sin(θ) + ds['Z']*np.cos(θ)).values),
#         'Xr_U': (['XG', 'Z'], (ds['XG']*np.cos(θ) - ds['Z']*np.sin(θ)).values),
#         'Zr_W': (['XC', 'Zl'], (ds['XC']*np.sin(θ) + ds['Zl']*np.cos(θ)).values),
#         'Xr_W': (['XC', 'Zl'], (ds['XC']*np.cos(θ) - ds['Zl']*np.sin(θ)).values)
#     })

#     coords = coords.assign_coords({
#         'Zr': (['XC', 'Z'], (coords['XC']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
#         'Xr': (['XC', 'Z'], (coords['XC']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
#         'Zr_V': (['XC', 'Z'], (coords['XC']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
#         'Xr_V': (['XC', 'Z'], (coords['XC']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
#         'Zr_U': (['XG', 'Z'], (coords['XG']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
#         'Xr_U': (['XG', 'Z'], (coords['XG']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
#         'Zr_W': (['XC', 'Zl'], (coords['XC']*np.sin(θ) + coords['Zl']*np.cos(θ)).values),
#         'Xr_W': (['XC', 'Zl'], (coords['XC']*np.cos(θ) - coords['Zl']*np.sin(θ)).values)
#     })
    
#     coords['Depthr'] = coords['Depth'] - coords['XC']*np.tan(θ)

#     return ds, coords

def add_background_temp(ds, Γ):
    ds['THETA'] = ds['THETA'].where(ds['THETA'] != 0.)
    ds['THETA_BG'] = ds['THETA'] + ds['Zr'] * Γ
    return ds

def add_Nsq(ds, Γ):
    ds['Nsq'] = -g/ρ0*ds['DRHODR']
    ds['Nsq'] = ds['Nsq'].where(ds['Nsq'] != 0.)
    ds['Nsq'] += g*α*Γ
    return ds
    
def add_hab_coordinates(ds, coords):
    
    ds = ds.assign_coords({
        'Z_hab':  ds['Z'] + (coords['Depth'] - coords['Depth'].max()),
        'Zl_hab': ds['Zl'] + (coords['Depth'] - coords['Depth'].max())
    })

    coords = coords.assign_coords({
        'Z_hab':  coords['Z'] + (coords['Depth'] - coords['Depth'].max()),
        'Zl_hab': coords['Zl'] + (coords['Depth'] - coords['Depth'].max())
    })
    return ds, coords

def _interp(x, y, bins=None):
    return np.interp(
        bins,
        x[::-1],
        y[::-1]
    )

def hab_interp(da, hab_bins=np.arange(2.5, 2000., 5.), vert_coord='Z'):
    hab = xr.DataArray(hab_bins, coords={"hab": hab_bins}, dims=["hab"])
    da_itp = xr.apply_ufunc(
        _interp,
        da[vert_coord+'_hab'],
        da,
        vectorize=True,
        dask='parallelized',
        input_core_dims=[[vert_coord], [vert_coord]],
        output_core_dims=[['hab']],
        output_sizes={'hab': hab.size},
        output_dtypes=[float],
        kwargs={'bins': hab_bins},
    )
    da_itp = da_itp.assign_coords({'hab': hab})
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

def periodic_extend(ds, concat_dims, dim_length, extend_multiples):
    ds_list = []
    for extend in range(extend_multiples[0], extend_multiples[1]+1):
        tmp = ds.copy()
        tmp_attrs = [tmp[dim].attrs for dim in concat_dims]
        for dim in concat_dims:
            tmp[dim] = tmp[dim] + extend*dim_length
        ds_list.append(tmp)
    ds = parallel_combine(ds_list, concat_dims = concat_dims)
    for i, dim in enumerate(concat_dims):
        ds[dim].attrs = tmp_attrs[i]
    return ds

def preprocess(ds, θ):
    H = ds['Depth'].max().values.copy().astype('>f8')
    ds['Z'] += H
    ds['Zl'] += H
    ds['Zu'] += H
    ds['Zp1'] += H

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


def tracer_flux_budget(ds, grid, suffix, θ=0., Γ=0.):
    """Calculate the convergence of fluxes of tracer `suffix` where
    `suffix` is `_TH`, `Tr01`, or 'Tr02'. Return a new xarray.Dataset."""
    new_suffix = suffix
    if new_suffix[0] != "_":
        new_suffix = "_"+new_suffix
    
    conv_horiz_adv_flux = -(grid.diff(ds['ADVx' + suffix], 'X') +
                          grid.diff(ds['ADVy' + suffix], 'Y')).rename('conv_horiz_adv_flux' + new_suffix)
    conv_horiz_diff_flux = -(grid.diff(ds['DFxE' + suffix], 'X') +
                          grid.diff(ds['DFyE' + suffix], 'Y')).rename('conv_horiz_diff_flux' + new_suffix)

    # sign convention is opposite for vertical fluxes
    conv_vert_adv_flux = (
        grid.diff(ds['ADVr' + suffix], 'Z', boundary='fill')
        .rename('conv_vert_adv_flux' + new_suffix)
    )
    conv_vert_diff_flux = (
        grid.diff(ds['DFrI' + suffix], 'Z', boundary='fill')
        .rename('conv_vert_diff_flux' + new_suffix)
    )

    all_fluxes = [
        conv_horiz_adv_flux, conv_horiz_diff_flux, conv_vert_adv_flux, conv_vert_diff_flux
    ]

    if suffix=="_TH":
        # anomalous fluxes
        conv_vert_diff_flux_anom = (-(grid.diff(
            ds['KVDIFF'].where(ds['WVEL'] != 0.), 'Z', boundary='fill'
        )/(ds['drF']*ds['hFacC'])*np.cos(θ)*Γ*ds['dV'])
        ).rename('conv_vert_diff_flux_anom' + new_suffix)
        conv_adv_flux_anom = -(
            grid.interp(ds['UVEL'], 'X')*Γ*np.sin(θ)*ds['dV'] +
            grid.interp(ds['WVEL'], 'Z', boundary='fill')*Γ*np.cos(θ)*ds['dV']
        ).rename('conv_adv_flux_anom' + new_suffix)
        all_fluxes += [conv_vert_diff_flux_anom, conv_adv_flux_anom]

    conv_all_fluxes = sum(all_fluxes).rename('conv_total_flux' + new_suffix)

    return xr.merge(all_fluxes + [conv_all_fluxes])