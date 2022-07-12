import numpy as np
import os

from landscape_utils import makeplot, save_dir, makeplot2

###### sheet landscapes ######
names = ['ico', 'icor', 'ico3', 'ico3r']
energies = {}
for name in names:
    energies[name] = np.load(os.path.join(save_dir, 'energies_%s.npy' % name))

vmin = np.amin([np.amin(x) for x in energies.values()])
vmax = np.amax([np.amax(x) for x in energies.values()])

for name in names:
    makeplot(n, energies[n], vmin=vmin, vmax=vmax)

###### combined landscapes ######

makeplot2('ico', 'icor', energies['ico'], energies['icor'])
makeplot2('ico3', 'ico3r', energies['ico3'], energies['ico3r'])

###### relaxed sheets ######

energies_rlx = {}
for name in names:
    energies_rlx[name] = np.load(os.path.join(save_dir, 'energies_%s.npy' % name))

vmin = np.amin([np.amin(x) for x in energies.values()])
vmax = np.amax([np.amax(x) for x in energies.values()])

for name in names:
    makeplot(n, energies[n], vmin=vmin, vmax=vmax)
