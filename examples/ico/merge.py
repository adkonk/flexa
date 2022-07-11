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
        return FlexaSheet.load(fpath, silent=1)

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

name = 'ico'
name_inv = 'icor'
v, f = ico(3, radius=2, angle=2*np.pi/5)

s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=phi0, psi0=psi0,
    ell0=1.5, normals='free', silent=1)
s_inv = invert_sheet(s)

# repeat for larger icosphere
name3 = 'ico3'
name3_inv = 'ico3r'
v, f = ico(3, radius=2, angle=3*np.pi/5) # greater angle for selection

s3 = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=phi0, psi0=psi0,
    ell0=1.5, normals='free', silent=1)
s3_inv = invert_sheet(s3)

# n = 10
# phis = phis[5:(5+n)]
# psis = psis[5:(5+n)]

ell0 = s.ell0 # ell0 is never changed to make s_inv or s3_inv
k = (1, 2, 10)

names = [name, name_inv, name3, name3_inv]
energies = {n: np.zeros((psis.size, phis.size)) for n in names}

energies[name][init_i, init_i] = initial_sim(s, name, ell0, k)
energies[name_inv][init_i, init_i] = initial_sim(s_inv, name_inv, ell0, k)
energies[name3][init_i, init_i] = initial_sim(s3, name3, ell0, k)
energies[name3_inv][init_i, init_i] = initial_sim(s3_inv, name3_inv, ell0, k)

paths = bfs_inds(energies[name])

pi = 0
n_sheets = energies[name].size
for p in paths:
    print('\nSheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (pi + 1, n_sheets, phis[p[1][0]], psis[p[1][1]]))
    pi += 1
    for n in names:
        print(n)
        s = sheet_from_path(p, n, phis, psis, ell0, k, save_dir)
        energies[n][p[1][1], p[1][0]] = s.energy(s.x, k)

def makeplot(name, energies, vmin=None, vmax=None):
    plt.figure(figsize=(12,12))
    meshplotter(phis, psis, energies,
        path_func=lambda phi, psi: file_path(save_dir, name, phi, psi, ell0, k),
        title='energies', cbarlabel='energy', log=True, vmin=vmin, vmax=vmax)
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

vmin = np.amin([np.amin(x) for x in energies.values()])
vmax = np.amax([np.amax(x) for x in energies.values()])

for n in names:
    makeplot(n, energies[n], vmin=vmin, vmax=vmax)

makeplot2('ico', 'icor', energies['ico'], energies['icor'])
makeplot2('ico3', 'ico3r', energies['ico3'], energies['ico3r'])
