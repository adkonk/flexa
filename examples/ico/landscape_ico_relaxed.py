import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from examples.utils import save_folder_path, file_path, bfs_inds, \
    meshplotter
from flexa._utils import picklify

from itertools import product

save_dir = save_folder_path('landscape', make=True)

def relax_sheet(name, phi, psi, ell0, k_orig, k_temp, dir_path=save_dir):
    fpath = file_path(dir_path, name, phi, psi, ell0, k_orig)
    # remove extension .p, add _rlx, add back .p
    savepath = picklify(os.path.splitext(fpath)[0] + '_rlx')
    if os.path.exists(savepath):
        return(FlexaSheet.load(savepath))

    s = FlexaSheet.load(fpath)
    s.f_equil(k_temp, tol=1e-5, rate=5e-4, plot=False, plotint=100, silent=1)
    s.f_equil(k_orig, tol=1e-5, rate=5e-4, plot=False, plotint=100, silent=1)
    s.save(savepath)
    return(s)

rang = 0.4
n = 21
phis = np.linspace(-rang, rang, n) + 0.55
psis = np.linspace(-rang, rang, n) + 0.65
ell0 = 0.5

k_orig = (1, 2, 10)
k_temp = (1, 2, 0.1)

name = 'ico'
name_inv = 'icor'
name3 = 'ico3'
name3_inv = 'ico3r'
names = [name, name_inv, name3, name3_inv]

energies = {n: np.zeros((psis.size, phis.size)) for n in names}

pi = 0
n_sheets = energies[name].size
for (phi_i, psi_i) in product(range(phis.size), range(psis.size)):
    print('\nSheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (pi + 1, n_sheets, phis[phi_i], psis[psi_i]))
    pi += 1
    for n in names:
        print('\n', n)
        s = relax_sheet(n, phis[phi_i], psis[psi_i], ell0, k_orig, k_temp, 
            save_dir)
        energies[n][psi_i, phi_i] = s.energy(s.x, k_orig)

def makeplot(name, energies):
    plt.figure(figsize=(12,12))
    meshplotter(phis, psis, energies, 
        path_func=lambda phi, psi: \
            file_path(save_dir, name, phi, psi, ell0, k_orig),
        title='energies', cbarlabel='energy', log=True)
    plt.savefig(os.path.join(save_dir, 'landscape_%s_rlx.png' % name), dpi=200)

for n in names:
    makeplot(n, energies[n])