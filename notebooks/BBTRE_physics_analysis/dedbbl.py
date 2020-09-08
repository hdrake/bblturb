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

def bbl_exp_spinup(
        T=10000*365*86400., dt=100*365*86400.,
        k0=5.2e-5, k1=1.8e-3, h=230.,
        N=1.3E-3, f=-5.3e-5, θ=1.26E-3, σ=1.,
        output_path = None,
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
    
    nu = domain.new_field(name='nu')
    nu['g'] = (k0+k1*np.exp(-z/h))*σ
    problem.parameters['nu'] = nu

    # Main equations
    problem.add_equation("-f*cos(θ)*v - sin(θ)*b - dz(nu*uz) = 0")
    problem.add_equation("f*u*cos(θ) - dz(nu*vz) = 0")
    problem.add_equation("dt(b) + N**2*sin(θ)*u - dz(k*bz) = N**2*cos(θ)*dz(k)")

    # Auxiliary equations defining the first-order reduction
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("vz - dz(v) = 0")
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
    if output_path is not None:
        analysis = solver.evaluator.add_file_handler(output_path, iter=1)

        # save buoyancy-budget diagnostics
        analysis.add_task("-N**2*sin(θ)*u", layout='g', name='advection')
        analysis.add_task("N**2*cos(θ)*dz(k) + dz(k*bz)", layout='g', name='diffusion')

        # save state variables
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