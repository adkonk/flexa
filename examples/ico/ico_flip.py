import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from examples.utils import save_folder_path

import copy

save_dir = save_folder_path('dynamics', make=True)

def invert_sheet(s):
    s_inv = copy.deepcopy(s)
    s_inv.x[:(3 * s_inv.n_cells)] *= 0.5 # move the cells inside the sheet
    s_inv.x[(3 * s_inv.n):] *= -1 # flip the normals
    s_inv.s0 = s_inv.sector_angles(s_inv.x.reshape(-1, 3)) # correct angles
    return(s_inv)

name = 'ico'
phi0 = 0.31
psi0 = 0.65

v, f = ico(3, radius=2, angle=2*np.pi/5)

s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=phi0, psi0=psi0, 
    ell0=1.5, normals='free', silent=1)
if name[-1] == 'r':
    s = invert_sheet(s)

ell0 = s.ell0
k_orig = (1, 2, 10)
k_temp = (1, 2, 0.1)

m = s.f_equil(k_orig, tol=1e-5, rate=5e-5, plot=True, plotint=50, silent=0, 
    plotdir=save_dir)
m = s.f_equil(k_temp, tol=1e-5, rate=5e-5, plot=True, plotint=50, silent=0, 
    plotdir=save_dir, m=m)
m = s.f_equil(k_orig, tol=1e-5, rate=5e-5, plot=True, plotint=50, silent=0, 
    plotdir=save_dir, m=m)