from xmitgcm import utils
import numpy as np

def add_attributes_from_datafile(ds, data_dir, datafiles=["data"]):
    for datafile in datafiles:
        dic = utils.parse_namelist(data_dir+datafile)
        for _,par in dic.items():
            for k,v in par.items():
                ds.attrs[k] = v

def barotropic_tidal_response(
        t, U0,
        ω=2*np.pi/(12.*3600.),
        ϕ=0.,
        f=0.,
        Γ=0.,
        θ=0.,
        α=2.e-4,
        g=9.81
    ):
    # dependent parameters
    N2 = α*g*Γ
    Ur = U0*ω**2/(ω**2-f**2-N2*np.sin(θ)**2)
    
    # wave solutions
    u = -Ur*(f/ω*np.sin(ϕ)*np.cos(ω*t) - np.cos(ϕ)*np.sin(ω*t))
    v = Ur*(f/ω*np.cos(ϕ)*np.cos(ω*t) + (1. - N2*np.sin(θ)**2/ω**2)*np.sin(ϕ)*np.sin(ω*t))
    T = Ur/(g*α)*N2*np.sin(θ)/ω*(np.cos(ϕ)*np.cos(ω*t) + f/ω*np.sin(ϕ)*np.sin(ω*t))
    return {"u": u, "v": v, "T": T}
