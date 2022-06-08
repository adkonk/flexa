import numpy as np
import matplotlib.pyplot as plt
import os

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from examples.utils import save_folder_path

save_dir = save_folder_path('plots')

# TODO: facegen gives an error when silent=0 and angle > 3*np.pi/5.
# This probably has to do with collar vertices overlapping with z positions
# as the icosphere curves radially in
# No errors when solving shape though!
v, f = ico(3, radius=2, angle=3*np.pi/5)

s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=0.654, psi0=0.8375, ell0=1, silent=1)
s.draw('3d')
plt.savefig(os.path.join(save_dir, 'init.png'), dpi=200)

s.solve_shape((1, 2, 10))
s.draw('3d')
plt.savefig(os.path.join(save_dir, 'final.png'), dpi=200)