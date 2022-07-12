import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from examples.utils import save_folder_path, file_path, bfs_inds, \
    meshplotter

import copy

save_dir = save_folder_path('landscape', make=True)

def initial_sim(s, name, ell0, k, dir_path=save_dir):
    print('Initial simulation for %s' % name)
    fpath = file_path(dir_path, name, s.phi0, s.psi0, ell0, k)
    if os.path.exists(fpath):
        s = FlexaSheet.load(fpath)
        s.f_equil(k, tol=1e-5, rate=5e-4, plot=False, plotint=1e3, silent=0)
        s.save(fpath)
        return s.energy(s.x, k)
    s.f_equil(k, tol=1e-5, rate=5e-4, plot=False, plotint=1e3, silent=0)
    s.save(fpath)
    return s.energy(s.x, k)

def sheet_from_path(path, name, phis, psis, ell0, k, dir_path=save_dir):
    fro = path[0] # from indices
    fpath0 = file_path(dir_path, name, phis[fro[0]], psis[fro[1]], ell0, k)
    to = path[1] # to indices
    fpath = file_path(dir_path, name, phis[to[0]], psis[to[1]], ell0, k)
    if os.path.exists(fpath):
        s = FlexaSheet.load(fpath, silent=1)
        s.f_equil(k, tol=1e-5, rate=5e-4, plot=False, plotint=100, silent=1)
        s.save(fpath)
        return s

    s = FlexaSheet.load(fpath0, silent=1)
    s.phi0 = phis[to[0]]
    s.psi0 = psis[to[1]]
    s.f_equil(k, tol=1e-5, rate=5e-4, plot=False, plotint=100, silent=1)
    s.save(fpath)
    return s

def invert_sheet(s):
    s_inv = copy.deepcopy(s)
    s_inv.x[:(3 * s_inv.n_cells)] *= 0.5 # move the cells inside the sheet
    s_inv.x[(3 * s_inv.n):] *= -1 # flip the normals
    s_inv.s0 = s_inv.sector_angles(s_inv.x.reshape(-1, 3)) # correct angles
    return s_inv

rang = 0.4
n = 21
init_i = round(n / 2)
phis = np.linspace(-rang, rang, n) + 0.55
psis = np.linspace(-rang, rang, n) + 0.65
phi0 = phis[init_i]
psi0 = psis[init_i]

k = (1, 2, 5)
ell0 = 0.5

energies = np.zeros((psis.size, phis.size))
paths = bfs_inds(energies)

def traverse_landscape(initial_sheet, name, c=0):
    if c == 0:
        initial_sim(initial_sheet, name, initial_sheet.ell0, k, save_dir)
        x = 1
        paths = []
        while init_i + x < n:
            paths = paths + [((init_i, init_i + x - 1), 
                (init_i, init_i + x))]
            paths = paths + [((init_i, init_i - x + 1), 
                (init_i, init_i - x))]  
            x += 1
    if c > 0:
        x = 1
        paths = []
        while init_i + x < n:
            paths = paths + [((init_i + x - 1, c), (init_i + x, c))]
            x += 1
    if c < 0:
        x = 1
        paths = []
        while init_i + x < n:
            paths = paths + [((init_i - x + 1, -c), (init_i - x, -c))]    
            x += 1

    pi = 0
    n_sheets = len(paths)
    for p in paths:
        print('\nSheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
            (pi + 1, n_sheets, phis[p[1][0]], psis[p[1][1]]))
        pi += 1
        s = sheet_from_path(p, name, phis, psis, ell0, k, save_dir)
        energies[p[1][1], p[1][0]] = s.energy(s.x, k)
    np.save(os.path.join(save_dir, 'energies_%s_%d.npy' % (name, c)))

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

def makeplot(name, energies):
    plt.figure(figsize=(12,12))
    meshplotter(phis, psis, energies,
        path_func=lambda phi, psi: file_path(save_dir, name, phi, psi, ell0, k),
        title='energies', cbarlabel='energy', log=True)
    plt.savefig(os.path.join(save_dir, 'landscape_%s.png' % name), dpi=200)

def makeplot2(name1, name2, energies1, energies2):
    plt.figure(figsize=(12,12))
    meshplotter(phis, psis,
        np.where(energies1 < energies2, energies1, energies2),
        path_func=lambda phi, psi: \
            file_path(save_dir, name1, phi, psi, ell0, k),
        title='energies', cbarlabel='energy', log=True)
    fig = plt.gcf()
    ax = fig.get_axes()[0]
    x, y = np.meshgrid(phis, psis)
    which = energies1 < energies2
    ax.plot(x[which], y[which], 'r.', ms=10)
    ax.plot(x[np.logical_not(which)], y[np.logical_not(which)], '.',
        color='orange', ms=10)
    plt.savefig(os.path.join(save_dir, 'landscape_%s_merge.png' % \
        (name1 + name2)), dpi=200)


