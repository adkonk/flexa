import numpy as np

from flexa.FlexaSheet import FlexaSheet
from flexa.lattice import ico
from landscape_utils import traverse_landscape, invert_sheet, phi0, psi0

name = 'ico3'

v, f = ico(3, radius=2, angle=3*np.pi/5)
s = FlexaSheet.facegen(v, f, z=0.5, ref='ori',
    phi0=phi0, psi0=psi0,
    normals='free', silent=1)

traverse_landscape(s, name)

