import flexa.lattice
from flexa.FlexaSheet import FlexaSheet

import matplotlib.pyplot as plt

points = flexa.lattice.random_sphere_points(100, radius = 1, angle=0)

s = FlexaSheet.vorgen(points, phi0=0, psi0=0, silent=1, z=0.3, ref = 'ori')
s.draw('3d')

plt.show()