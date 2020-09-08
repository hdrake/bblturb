import numpy as np
import h5py
import xarray as xr

def itp(var):
    return 0.5*(var[1:] + var[:-1])

def sigmoid(x, x0=760., Lx=175.):
    return 1/(1+np.exp(-(x-x0)/Lx))

def calc_S(N, θ, f): return N**2*np.tan(θ)**2/f**2;
def calc_q(N, θ, f, k, σ=1.):
    if f==0.:
        return ((np.cos(θ)**2*N**2*np.tan(θ)**2*σ)/(4*σ**2*k**2))**(1/4);
    else:
        return (f**2*np.cos(θ)**2*(1+calc_S(N,θ,f)*σ)/(4*σ**2*k**2))**(1/4);
def calc_δ(N, θ, f, k, σ=1.): return 1/calc_q(N,θ,f,k,σ)

def k_exp(k0, k1, h, z):
    return k0+k1*np.exp(-z/h)

def bbl_exp(k0, k1, h, N, f, θ, σ=1., H=2500., dz=1.):
    if f==0.:
        Sfac = 1.;
    else:
        S = calc_S(N, θ, f)
        Sfac = S*σ/(1+S*σ)
    q = calc_q(N, θ, f, k0+k1, σ)
    z = np.arange(dz/2., H+dz/2., dz)
    u = -k1*np.tan(θ)**-1*np.exp(-z/h)/h*Sfac + \
        2.*q*np.tan(θ)**-1*(k0+k1*Sfac) * \
        np.exp(-q*z)*np.sin(q*z)
    zf = np.arange(0., H+dz, dz)
    bz = N**2*np.cos(θ)*(
        k0/(k0+k1*np.exp(-zf/h)) +
        k1*np.exp(-zf/h)/(k0+k1*np.exp(-zf/h))*Sfac -
        (k0/(k0+k1) + k1/(k0+k1)*Sfac) *
        np.exp(-q*zf)*(np.cos(q*zf) + np.sin(q*zf))
    )
    vz = f*np.tan(θ)**-1*np.cos(θ)/σ * (
        k1*np.exp(-zf/h)/(k0+k1*np.exp(-zf/h))*Sfac -
        (k0/(k0+k1) + k1/(k0+k1)*Sfac) *
        np.exp(-q*zf)*(np.cos(q*zf) + np.sin(q*zf))
    )
    v = np.cumsum(vz*dz)
    v = v-v[0]
    return {"z":z, "zf":zf, "bz":bz, "u":u, "v":v, "vz":vz}

def bbl_to_ds(bbl):
    ds = xr.Dataset()
    ds = ds.assign_coords({'Z': bbl['z'], 'Zl': bbl['zf']})
    mapping = {'bz': 'Zl', 'vz': 'Zl', 'b': 'Z', 'u': 'Z', 'v': 'Z'}
    for k,v in mapping.items():
        ds[k] = xr.DataArray(bbl[k], coords=(ds.coords[v],), dims=v)
    return ds