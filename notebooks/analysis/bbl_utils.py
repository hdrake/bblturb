from xgcm import Grid
import numpy as np

def z_mean_profile(da):
    da_m = da.where(da != 0.)
    return da_m.mean(dim=['XC', 'YC'])

def weighted_mean(da, weights, **kwargs):
    return (da*weights).sum(**kwargs)/weights.sum(**kwargs)

def pre_process(ds, timestep=60.):
    ds['time'] = ds['time']*timestep # convert time from iterations to seconds
    
    coords = ds.coords.to_dataset().reset_coords()
    ds = ds.reset_coords(drop=True)
    
    # Add grid metrics
    coords['drW'] = coords.hFacW * coords.drF #vertical cell size at u point
    coords['drS'] = coords.hFacS * coords.drF #vertical cell size at v point
    coords['drC'] = coords.hFacC * coords.drF #vertical cell size at tracer point
    metrics = {
        ('X',): ['dxC', 'dxG'], # X distances
        ('Y',): ['dyC', 'dyG'], # Y distances
        ('Z',): ['drW', 'drS'], # Z distances
        ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
    }
    grid = Grid(coords, metrics=metrics, periodic=['X','Y'])
    # Add interpolated grid metrics
    coords['dxF'] = grid.interp(coords.dxC,'X')
    coords['dyF'] = grid.interp(coords.dyC,'Y')
    metrics = {
        ('X',): ['dxC', 'dxG', 'dxF'], # X distances
        ('Y',): ['dyC', 'dyG', 'dyF'], # Y distances
        ('Z',): ['drW', 'drS', 'drC'], # Z distances
        ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
    }
    grid = Grid(coords, metrics=metrics, periodic=['X','Y'])
    
    H = coords['Depth'].max().values.copy().astype('>f8')
    ds = ds.assign_coords(
        Z = ds['Z'] + H,
        Zl = ds['Zl'] + H,
        Zu = ds['Zu'] + H,
        Zp1 = ds['Zp1'] + H
    )
    coords = coords.assign_coords(
        Z = coords['Z'] + H,
        Zl = coords['Zl'] + H,
        Zu = coords['Zu'] + H,
        Zp1 = coords['Zp1'] + H
    )

    return ds, coords, grid


def add_cartesian_coordinates(ds, coords, θ):
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

    coords = coords.assign_coords({
        'Zr': (['XC', 'Z'], (coords['XC']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
        'Xr': (['XC', 'Z'], (coords['XC']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
        'Zr_V': (['XC', 'Z'], (coords['XC']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
        'Xr_V': (['XC', 'Z'], (coords['XC']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
        'Zr_U': (['XG', 'Z'], (coords['XG']*np.sin(θ) + coords['Z']*np.cos(θ)).values),
        'Xr_U': (['XG', 'Z'], (coords['XG']*np.cos(θ) - coords['Z']*np.sin(θ)).values),
        'Zr_W': (['XC', 'Zl'], (coords['XC']*np.sin(θ) + coords['Zl']*np.cos(θ)).values),
        'Xr_W': (['XC', 'Zl'], (coords['XC']*np.cos(θ) - coords['Zl']*np.sin(θ)).values)
    })

    return ds, coords

def add_background_temp(ds, Γ):
    ds['THETA'] = ds['THETA'].where(ds['THETA'] != 0.)
    ds['THETA_BG'] = ds['THETA'] + ds['Zr'] * Γ
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

def interp_to_hab_coords(ds):
    return