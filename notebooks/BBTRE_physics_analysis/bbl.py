import numpy as np
import h5py
import xarray as xr
from dedalus import public as de

import warnings
warnings.filterwarnings("ignore")

import logging
root = logging.root
for hand in root.handlers:
    hand.setLevel("INFO")
logger = logging.getLogger(__name__)

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

def bbl_exp(k0, k1, h, N, f, θ, σ=1.):
    if f==0.:
        Sfac = 1.;
    else:
        S = calc_S(N, θ, f)
        Sfac = S*σ/(1+S*σ)
    q = calc_q(N, θ, f, k0+k1, σ)
    dz = 1.
    z = np.arange(dz/2, 2500.+dz, dz)
    bz = N**2*np.cos(θ)*(
        k0/(k0+k1*np.exp(-z/h)) +
        k1*np.exp(-z/h)/(k0+k1*np.exp(-z/h))*Sfac -
        (k0/(k0+k1) + k1/(k0+k1)*Sfac) *
        np.exp(-q*z)*(np.cos(q*z) + np.sin(q*z))
    )
    u = -k1*np.tan(θ)**-1*np.exp(-z/h)/h*Sfac + \
        2.*q*np.tan(θ)**-1*(k0+k1*Sfac) * \
        np.exp(-q*z)*np.sin(q*z)
    zf = np.arange(0., 2500.+dz, dz)
    vz = f*np.tan(θ)**-1*np.cos(θ)/σ * (
        k1*np.exp(-zf/h)/(k0+k1*np.exp(-zf/h))*Sfac -
        (k0/(k0+k1) + k1/(k0+k1)*Sfac) *
        np.exp(-q*zf)*(np.cos(q*zf) + np.sin(q*zf))
    )
    v = np.cumsum(vz*dz)
    v = v-v[0]
    return {"z":z, "zf":zf, "bz":bz, "u":u, "v":v, "vz":vz}

def bbl_exp_spinup(
        T=10000*365*86400., dt=100*365*86400.,
        k0=5.2e-5, k1=1.8e-3, h=230.,
        N=1.3E-3, f=-5.3e-5, θ=1.26E-3, σ=1.,
        #area_weight=False
    ):

    #===== Set up domain =====
    # Create basis and domain for all experiments
    Lz = 2700
    nz = 256
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
    domain = de.Domain([z_basis],np.float64)
    z = domain.grid(0)

    #===== Set up σoblem ======
    problem = de.IVP(domain, variables=['u', 'v', 'b', 'uz', 'vz', 'bz']);

    # Set parameters
    problem.parameters['f'] = f
    problem.parameters['θ'] = θ # set in loop
    problem.parameters['N'] = N
    k = domain.new_field(name='k')
    k['g'] = k0+k1*np.exp(-z/h)
    problem.parameters['k'] = k
    
#     if area_weight:
#         a = domain.new_field(name='a')
#         a['g'] = sigmoid(z, 760., 175.)
#         problem.parameters['a'] = a
#         ar = domain.new_field(name='ar')
#         ar['g'] = 1./sigmoid(z, 760., 175.)
#         problem.parameters['ar'] = ar
#     else:
#         problem.parameters['a'] = 1.
#         problem.parameters['ar'] = 1.

    nu = domain.new_field(name='nu')
    nu['g'] = (k0+k1*np.exp(-z/h))*σ
    problem.parameters['nu'] = nu

    # Main equations
    problem.add_equation("-f*cos(θ)*v - sin(θ)*b - dz(nu*uz) = 0")
    problem.add_equation("f*u*cos(θ) - dz(nu*vz) = 0")
    problem.add_equation("dt(b) + N**2*sin(θ)*u - dz(k*bz) = N**2*cos(θ)*dz(k)")
    #problem.add_equation("dt(b) + N**2*sin(θ)*u - ar*dz(k*a*bz) = ar*N**2*cos(θ)*dz(a*k)")

    # Auxiliary equations defining the first-order reduction
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("vz - dz(v) = 0")
    #problem.add_equation("bz - ar*dz(a*b) = 0")
    problem.add_equation("bz - dz(b) = 0")

    # Boundary conditions
    problem.add_bc('left(u) = 0')
    problem.add_bc('left(v) = 0')
    problem.add_bc('left(bz) = - N**2*cos(θ)')
    problem.add_bc('right(uz) = 0')
    problem.add_bc('right(vz) = 0')
    problem.add_bc('right(bz) = 0')

    # Set solver for IVP
    solver = problem.build_solver(de.timesteppers.RK443);

    #==== Set initial conditions ====
    # Reference local grid and state fields
    z = domain.grid(0)
    u = solver.state['u']
    v = solver.state['v']
    b = solver.state['b']
    uz = solver.state['uz']
    vz = solver.state['vz']
    bz = solver.state['bz']

    # State from a state of rest
    u['g'] = np.zeros_like(z)
    v['g'] = np.zeros_like(z)
    b['g'] = np.zeros_like(z)
    u.differentiate('z', out=uz)
    v.differentiate('z', out=vz)
    b.differentiate('z', out=bz)

    #==== Create analysis files ====
    output_path = '../../data/dedalus/spinup'
    analysis = solver.evaluator.add_file_handler(output_path, iter=1)
    analysis.add_system(solver.state, layout='g')
    
    # Start / stop criteria
    solver.stop_sim_time = T
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Main loop
    year = 365.*24*60*60
    while solver.ok:
        solver.step(dt);

    z_da = domain.grid(0, scales=domain.dealias)
    
    output = {
        'z':z_da, 'zf': itp(z_da),
        'bz': itp(bz['g'])+N**2, 'vz': itp(vz['g']),
        'b': b['g'], 'u': u['g'], 'v': v['g']
    }
    
    return output

def bbl_to_ds(bbl):
    ds = xr.Dataset()
    ds = ds.assign_coords({'Z': bbl['z'], 'Zl': bbl['zf']})
    mapping = {'bz': 'Zl', 'vz': 'Zl', 'b': 'Z', 'u': 'Z', 'v': 'Z'}
    for k,v in mapping.items():
        ds[k] = xr.DataArray(bbl[k], coords=(ds.coords[v],), dims=v)
    return ds