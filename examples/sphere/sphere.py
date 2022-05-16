import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
import flexa.lattice
from examples.flat.utils_flat import save_folder_path, file_path, spiral_inds, \
    meshplotter

import copy

save_dir = save_folder_path('landscape', make=True)

def get_sheet(s, name, phi0, psi0, ell0, k, dir_path = save_dir):
    fpath = file_path(save_dir, name, phi0, psi0, ell0, k)
    if os.path.exists(fpath):
        s = FlexaSheet.load(fpath, silent=1)
    else: 
        s.phi0 = phi0
        s.psi0 = psi0
        s.solve_shape(k, silent=1)
        s.save(fpath)
    return(s)

name = 'sph'
np.random.seed(3)
points = flexa.lattice.random_sphere_points(30, radius = 1, angle=np.pi/3)
# phi0, psi0 = average phi, psi
s = FlexaSheet.vorgen(points, silent=0, z=0.3, ref = 'ori')

range = 0.4
n = 21
phis = np.linspace(-range, range, n) + s.phi0
psis = np.linspace(-range, range, n) + s.psi0

# n = 16
# phis = phis[3:(3+n)]
# psis = psis[2:(2+n)]

ell0 = np.mean(s.ell0)
k = 10

energies = np.zeros((n, n))
inds = spiral_inds(energies)[::-1, :] # spiral order starting from center

s_init = get_sheet(s, name, phis[inds[0, 0]], psis[inds[0, 1]], ell0, k)
s_init.draw('3d')
plt.show()

#good = np.zeros((n,n))

for ri in np.arange(inds.shape[0]):
    phi_i, psi_i = inds[ri, :]
    print('Sheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (ri + 1, inds.shape[0], phis[phi_i], psis[psi_i]))
    
    s = get_sheet(s_init, name, phis[phi_i], psis[psi_i], ell0, k)
    #s.draw('3d')
    #plt.show()
    #good[phi_i, psi_i] = int(input('good? '))
    #np.save('good.npy', good)
    energies[phi_i, psi_i] = s.energy(s.x, k)
    print(energies[phi_i, psi_i])

np.save('energies.npy', energies)

plt.figure(figsize=(12,12))
meshplotter(phis, psis, energies, 
    path_func=lambda phi, psi: file_path(save_dir, name, phi, psi, ell0, k),
    title='energies', cbarlabel='energy', log=True)
plt.savefig(os.path.join(save_dir, 'landscape_sph.png'), dpi=200)
