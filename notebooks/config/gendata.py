import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.fftpack import fftn, ifftn

perturbations = True
rotatedFrame = True
noisy = False
hills = False

plotting = True

# create grid
nx = 128
ny = 128
nz = 300

dx = 300
dy = 300
dz0 = 3
nz_vary = np.int(nz*0.65)
dz = dz0 * np.ones((nz))[:,np.newaxis,np.newaxis]*np.ones((nz,ny,nx))
dz[nz_vary:,:,:] = dz[nz_vary,0,0]*1.0275**np.arange(0,nz-nz_vary,1.0)[:,np.newaxis,np.newaxis]*np.ones((nz-nz_vary,ny,nx))

Lx = nx*dx
Ly = ny*dy
Hz = sum(dz[:,0,0])

x = np.arange(dx/2.0,Lx,dx)[np.newaxis,np.newaxis,:]*np.ones((nz,ny,nx))
y = np.arange(dy/2.0,Ly,dy)[np.newaxis,:,np.newaxis]*np.ones((nz,ny,nx))
z = (-Hz + np.cumsum(dz,axis=0) - dz/2.0)

if plotting:
    plt.figure()
    plt.plot(dz[:,0,0],z[:,0,0],"r-+")
    plt.show()

# Create topography
slopeAngle = 2.e-3
if rotatedFrame: mean_slope = 0.
else: mean_slope = np.tan(slopeAngle)

print("Slope angle is: "+np.str(slopeAngle))

Hbot = np.zeros_like(x)
Hbot[0,:,:] = x[0,:,:]*mean_slope

if hills: Hbot += 50.*(1. + np.sin(3*(2*np.pi*x/Lx)))

Hbot = Hbot - (Hz + 2*dz0)
if not(rotatedFrame): Hbot[0,:,0] = 0; Hbot[0,:,-1] = 0;

if plotting:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.pcolor(x[0,:,:]*1e-3,y[0,:,:]*1e-3,Hbot[0,:,:])
    plt.xlabel('zonal distance [km]')
    plt.ylabel('meridional distance [km]')
    plt.colorbar()
    plt.clim([np.min(z),np.min(z)+900])

    plt.subplot(1,2,2)
    plt.plot(x[0,0,:]*1e-3,Hbot[0,ny//2,:])
    plt.xlim([0,nx*dx*1e-3])
    plt.xlabel('zonal distance [km]')
    plt.ylabel('depth [m]')

    plt.tight_layout()
    plt.show()

# generate initial conditions and temperature restoring
N = 1.3e-3
tAlpha = 2.e-4
gravity = 9.81
gamma = (N**2)/(gravity*tAlpha)
gamma_eps = gamma*1.e-4

U = np.zeros((nz,ny,nx))
V = np.zeros((nz,ny,nx))
if not(perturbations): T = gamma * (z+Hz);
else: T = np.zeros((nz,ny,nx)) + gamma_eps * (z+Hz)

if noisy:
    T += 1.e-16*(np.random.random((nz,ny,nx))-0.5)

print("Temperature Lapse Rate is: "+np.str((N**2)/(gravity*tAlpha)))

# generate 3D vertical eddy diffusivity field
d = 230
k0 = 5.2e-5
k1 = 1.8e-3
K = np.zeros((nz,ny,nx))
for i in range(nx):
    for j in range(ny):
        K[:, j, i] = k0 + k1*np.exp(-(z[:,0,0] - Hbot[0,j,i])/d)
K[K>(k1+k0)] = k1+k0

if plotting:
    plt.figure(figsize=(12,5))
    plt.pcolor(x[:,0,:]*1e-3,z[:,0,:],K[:,0,:])
    plt.xlabel('zonal distance [km]')
    plt.ylabel('depth [m]')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# Reverse vertical axis so first index is at the surface and transpose axes
U = U[::-1,:,:]
V = V[::-1,:,:]
T = T[::-1,:,:]
K = K[::-1,:,:]
dz = dz[::-1,:1,:1]

# save input data as binary files
newFile = open("U.init", "wb")
newFile.write(bytes(U.astype('>f8')))
newFile.close()

newFile = open("V.init", "wb")
newFile.write(bytes(V.astype('>f8')))
newFile.close()

newFile = open("T.init", "wb")
newFile.write(bytes(T.astype('>f8')))
newFile.close()

newFile = open("kappa.init", "wb")
newFile.write(bytes(K.astype('>f8')))
newFile.close()

newFile = open("topog.init", "wb")
newFile.write(bytes(Hbot[0,:,:].astype('>f8')))
newFile.close()

newFile = open("delZ.init", "wb")
newFile.write(bytes(dz[:,0,0].astype('>f8')))
newFile.close()

