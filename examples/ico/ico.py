from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico

import numpy as np
import matplotlib.pyplot as plt

import os

def save_folder_path(folder_name, make=True):
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(parent_dir, folder_name)
    if make and not os.path.isdir(path):
        os.mkdir(path)
    return(path)

path = save_folder_path('plots')

# TODO: facegen gives an error when silent=0 and angle > 3*np.pi/5.
# This probably has to do with collar vertices overlapping with z positions
# as the icosphere curves radially in
# No errors when solving shape though!
v, f = ico(2, radius=2, angle=3*np.pi/5)

s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=0.654, psi0=0.8375, ell0=1, silent=1)
plt.figure(figsize=(20,10))
s.draw('3d')
plt.savefig(path + '/init.png', dpi=200)

s.solve_shape(10)
plt.figure(figsize=(20,10))
s.draw('3d')
plt.savefig(path + '/final.png', dpi=200)