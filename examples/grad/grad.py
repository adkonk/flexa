import numpy as np
import matplotlib.pyplot as plt
from flexa.FlexaSheet import FlexaSheet
import os

from flexa.lattice import ico

plot = False

dir_path = os.path.dirname(os.path.realpath(__file__))
if plot:
    plotdir = os.path.join(dir_path, 'movie')
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
else: 
    plotdir = None

v, f = ico(2, radius=2, angle=3*np.pi/5)
s = FlexaSheet.facegen(v, f, z=0.5, ref='ori', ell0=2, silent=0)

plt.figure(figsize=(16, 8))
s.draw('3d')
plt.savefig(os.path.join(dir_path, 'init.png'), dpi=200)

s.f_equil((1, 2, 10), tol=2e-4, rate=1e-3, 
    plot=plot, plotdir=plotdir, plotint=10, 
    silent=0)

plt.figure(figsize=(16, 8))
s.draw('3d')
plt.savefig(os.path.join(dir_path, 'final.png'), dpi=200)