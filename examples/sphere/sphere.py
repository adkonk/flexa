import numpy as np
import matplotlib.pyplot as plt
import os

import flexa.lattice
from flexa.FlexaSheet import FlexaSheet
from examples.utils import save_folder_path, file_path, meshplotter

save_dir = save_folder_path('plots')
np.random.seed(3)

name = 'sph'
points = flexa.lattice.random_sphere_points(50, radius = 1, angle=np.pi/3)

# phi0, psi0 = average phi, psi
s = FlexaSheet.vorgen(points, silent=0, z=0.3, ref = 'ori')
s.draw('3d')
plt.savefig(os.path.join(save_dir, 'init.png'), dpi=200)

s.solve_shape((1, 2, 10))
s.draw('3d')
plt.savefig(os.path.join(save_dir, 'final.png'), dpi=200)