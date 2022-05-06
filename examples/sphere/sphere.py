import flexa.lattice
from flexa.FlexaSheet import FlexaSheet

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

np.random.seed(3)

points = flexa.lattice.random_sphere_points(50, radius = 1, angle=np.pi/3)

# phi0, psi0 = average phi, psi
s = FlexaSheet.vorgen(points, silent=0, z=0.3, ref = 'ori')

s.draw('3d')
plt.savefig(path + '/init.png', dpi=200)

s.solve_shape(10)
s.draw('3d')
plt.savefig(path + '/final.png', dpi=200)