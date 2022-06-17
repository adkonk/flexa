import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from examples.utils import save_folder_path, file_path, spiral_inds, \
    meshplotter

import copy

save_dir = save_folder_path('landscape', make=True)

def get_sheet(s, name, phi0, psi0, ell0, k, dir_path = save_dir):
    fpath = file_path(dir_path, name, phi0, psi0, ell0, k)
    if os.path.exists(fpath):
        s = FlexaSheet.load(fpath, silent=1)
    else: 
        s.phi0 = phi0
        s.psi0 = psi0
        s.f_equil(k, tol=5e-5, rate=1e-5, plot=False, silent=1)
        s.save(fpath)
    return(s)

name = 'icot'
name_inv = 'icotr'
v, f = ico(3, radius=2, angle=3*np.pi/5)

s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    # phi0=0.654, psi0=0.8375, 
    ell0=2, normals='free', silent=1)

s_inv = copy.deepcopy(s)
s_inv.x[:(3 * s_inv.n_cells)] *= 0.5 # move the cells inside the sheet
s_inv.x[(3 * s_inv.n):] *= -1

range = 0.4
n = 21
phis = np.linspace(-range, range, n) + s.phi0
psis = np.linspace(-range, range, n) + s.psi0

# n = 10
# phis = phis[5:(5+n)]
# psis = psis[5:(5+n)]

ell0 = s.ell0
k = (1, 2, 10)

energies = np.zeros((psis.size, phis.size))
energies_inv = np.zeros((psis.size, phis.size))
inds = spiral_inds(energies)[::-1, :] # spiral order starting from center

for ri in np.arange(inds.shape[0]):
    phi_i, psi_i = inds[ri, :]
    print('Sheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (ri + 1, inds.shape[0], phis[phi_i], psis[psi_i]))
    
    s = get_sheet(s, name, phis[phi_i], psis[psi_i], ell0, k)
    s_inv = get_sheet(s_inv, name_inv, phis[phi_i], psis[psi_i], ell0, k)
    
    energies[psi_i, phi_i] = s.energy(s.x, k)
    energies_inv[psi_i, phi_i] = s_inv.energy(s_inv.x, k)

plt.figure(figsize=(12,12))
meshplotter(phis, psis, energies, 
    path_func=lambda phi, psi: file_path(save_dir, name, phi, psi, ell0, k),
    title='energies', cbarlabel='energy', log=True)
plt.savefig(os.path.join(save_dir, 'landscape_icot.png'), dpi=200)

plt.figure(figsize=(12,12))
meshplotter(phis, psis, energies_inv, 
    path_func=lambda phi, psi: file_path(save_dir, name_inv, phi, psi, ell0, k),
    title='energies', cbarlabel='energy', log=True)
plt.savefig(os.path.join(save_dir, 'landscape_icotr.png'), dpi=200)