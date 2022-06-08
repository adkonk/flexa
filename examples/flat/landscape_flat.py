import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from examples.utils import save_folder_path, file_path, name_to_func, \
    spiral_inds, meshplotter

save_dir = save_folder_path('landscape', make=True)

def get_sheet(s, name, phi0, psi0, ell0, k, dir_path=save_dir):
    fpath = file_path(save_dir, name, phi0, psi0, ell0, k)
    if os.path.exists(fpath):
        s = FlexaSheet.load(fpath, silent=1)
    else: 
        s.phi0 = phi0
        s.psi0 = psi0
        s.solve_shape(k, silent=1)
        s.save(fpath)
    return(s)

name = 'hex'
f = name_to_func[name]
lattice = f()
s = FlexaSheet.flatgen(lattice, silent=1)

range = 0.2
n = 11
phis = np.linspace(-range, range, n) + s.phi0
psis = np.linspace(-range, range, n) + s.psi0

ell0 = s.ell0[0]
k = (1, 2, 10)

energies = np.zeros((n, n))
inds = spiral_inds(energies)[::-1, :] # spiral order starting from center

for ri in np.arange(inds.shape[0]):
    phi_i, psi_i = inds[ri, :]
    print('Sheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (ri + 1, inds.shape[0], phis[phi_i], psis[psi_i]))
    
    s = get_sheet(s, name, phis[phi_i], psis[psi_i], ell0, k)
    energies[phi_i, psi_i] = s.energy(s.x, k)

plt.figure(figsize=(12,12))
meshplotter(phis, psis, energies, 
    path_func=lambda phi, psi: file_path(save_dir, name, phi, psi, ell0, k),
    title='energies', cbarlabel='energy', log=True, vmin=1e-6)
plt.savefig(os.path.join(save_dir, 'landscape.png'), dpi=200)