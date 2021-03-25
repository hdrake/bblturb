from xgcm import Grid
import xarray as xr
import numpy as np

def add_gradients(ds, grid, v):
    tmp = grid.diff(ds[v], 'X', boundary='periodic')
    ds[f'd{v}dx'] = grid.interp((
            tmp.where(~np.isnan(tmp), 0.).chunk({'XG': 400, 'YC':-1}) /
            ds['dxC']
        ), 'X', boundary='periodic'
    ).chunk({'XC': 400, 'YC':-1})

    tmp = grid.diff(ds[v], 'Y', boundary='periodic')
    ds[f'd{v}dy'] = grid.interp((
            tmp.where(~np.isnan(tmp), 0.).chunk({'XC': 400, 'YG':-1}) /
            ds['dyC']
        ), 'Y', boundary='periodic'
    ).chunk({'XC': 400, 'YC':-1})

    tmp = grid.diff(ds[v], 'Z', boundary='fill', fill_value = np.nan)
    ds[f'd{v}dz'] = grid.interp((
            tmp.where(~np.isnan(tmp), 0.).chunk({'XC': 400, 'Zl': -1}) /
            grid.diff(ds['Z'], 'Z', boundary='extend')
        ), 'Z', boundary='extend'
    )

def kvdiff_mask(ds, grid):
    maskUp = grid.interp(ds['maskC'].astype('float64'), 'Z', boundary='extend') == 1.
    ds['KVDIFF'] = ds['KVDIFF'].where(maskUp, 0.)
    return ds


def tracer_flux_budget(ds, grid, suffix, θ=0., Γ=0., add_standing=False):
    """Calculate the convergence of fluxes of tracer `suffix` where
    `suffix` is `_TH`, `Tr01`, or 'Tr02'. Return a new xarray.Dataset."""
    new_suffix = suffix
    if new_suffix[0] != "_":
        new_suffix = "_"+new_suffix
    
    conv_horiz_adv_flux = (
        -(grid.diff(ds['ADVx' + suffix], 'X') +
          grid.diff(ds['ADVy' + suffix], 'Y'))
    ).rename('conv_horiz_adv_flux' + new_suffix)
    conv_horiz_diff_flux = (
        -(grid.diff(ds['DFxE' + suffix], 'X') +
          grid.diff(ds['DFyE' + suffix], 'Y'))
    ).rename('conv_horiz_diff_flux' + new_suffix)

    # sign convention is opposite for vertical fluxes
    conv_vert_adv_flux = (
        grid.diff(ds['ADVr' + suffix], 'Z', boundary='fill')
    ).rename('conv_vert_adv_flux' + new_suffix)
    conv_vert_diff_flux = (
        grid.diff(ds['DFrI' + suffix], 'Z', boundary='fill')
    ).rename('conv_vert_diff_flux' + new_suffix)

    all_fluxes = [
        conv_horiz_adv_flux, conv_horiz_diff_flux, conv_vert_adv_flux, conv_vert_diff_flux
    ]

    if suffix=="_TH":
        # mask KVDIFF
        ds = kvdiff_mask(ds, grid)
        
        # diffusion of mean buoyancy
        conv_vert_diff_flux_anom = (-(grid.diff(
            ds['KVDIFF'], 'Z', boundary='fill'
        )/(ds['drF']*ds['hFacC'])*np.cos(θ)*Γ*ds['dV'])
        ).rename('conv_vert_diff_flux_anom' + new_suffix)
        
        # perturbation advection of mean buoyancy
        conv_adv_flux_anom = (-(
            grid.interp(ds['UVEL'], 'X')*Γ*np.sin(θ)*ds['dV'] +
            grid.interp(ds['WVEL'], 'Z', boundary='fill')*Γ*np.cos(θ)*ds['dV']
        )).rename('conv_adv_flux_anom' + new_suffix)
        
        all_fluxes += [conv_vert_diff_flux_anom, conv_adv_flux_anom]

    conv_all_fluxes = sum(all_fluxes).rename('conv_total_flux' + new_suffix)
    
    if add_standing:
        conv_vert_adv_standing_flux = (
            grid.diff(ds['ADVr' + suffix + "_standing"], 'Z', boundary='fill')
        ).rename('conv_vert_adv_standing_flux' + new_suffix)
        all_fluxes.append(conv_vert_adv_standing_flux)

    return xr.merge(all_fluxes + [conv_all_fluxes])

def add_temp_budget(temp, grid, Γ, θ, add_standing=False):
    day2seconds = 1./(86400.)
    temp['dV'] = (temp.drF * temp.rA * temp.hFacC)
    
    budget = tracer_flux_budget(temp, grid, '_TH', Γ=Γ, θ=θ, add_standing=add_standing).chunk({'Z': -1, 'YC': -1, 'XC': 400})
    budget['total_tendency_TH'] = budget['conv_total_flux_TH']
    budget['total_tendency_TH_truth'] = temp.TOTTTEND * temp['dV'] * day2seconds
    budget['diff_tendency_TH'] = (budget['conv_horiz_diff_flux_TH'] + budget['conv_vert_diff_flux_TH'] + budget['conv_vert_diff_flux_anom_TH'])
    budget['adv_tendency_TH'] = (budget['conv_horiz_adv_flux_TH'] + budget['conv_vert_adv_flux_TH'] + budget['conv_adv_flux_anom_TH'])
    tmp = xr.merge([temp, budget])
    tmp.attrs = temp.attrs
    return tmp

def check_temp_budget_closes(temp, rel_err_tol=1.e-3):
    temp_int = temp[['total_tendency_TH', 'total_tendency_TH_truth']].sum(dim=['Z','YC', 'XC']).mean(dim='time').load()
    a, b = temp_int['total_tendency_TH'].values, temp_int['total_tendency_TH_truth'].values
    rel_err = np.abs((a-b)/((a+b)/2.))
    return rel_err < rel_err_tol