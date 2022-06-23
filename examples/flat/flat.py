import numpy as np
import matplotlib.pyplot as plt
import os

from collections.abc import Iterable
from flexa.FlexaSheet import FlexaSheet
from examples.utils import name_to_func, nameroot, save_folder_path

save_dir = save_folder_path('sheetplots', make=True)

def make_graphs(name, params):
    # params takes form [(phi0, psi0, ell0, k)] 
    # list of lists of parameters
    f = name_to_func[name]
    lattice = f()

    # make sure params is list of lists
    if not isinstance(params[0], Iterable):
        params = [params]

    s = FlexaSheet.flatgen(lattice, phi0=params[0][0], 
        psi0=params[0][1], ell0=params[0][2])
    plt.figure(figsize=(10,10))
    s.draw()
    plt.savefig('%s_graph.png' % name, dpi=200)

    for (phi0, psi0, ell0, k) in params:
        s.phi0 = phi0 
        s.psi0 = psi0
        s.ell0 = ell0
        s.f_equil(k, tol=5e-5, rate=5e-4, plot=False, plotint=200, silent=1)
        
        r = nameroot(name, phi0, psi0, ell0, k)
        
        plt.figure(figsize=(16,16))
        s.draw()
        plt.savefig(os.path.join(save_dir, '%s_graph.png' % r), dpi=200)
        
        plt.figure(figsize=(20,8))
        s.draw(style='3d')
        plt.savefig(os.path.join(save_dir, '%s_plot.png' % r), dpi=200)

params = [[0.8, 0.8, 1.52, (1, 2, 10)], [0.95, 0.8, 1.52, (1, 2, 10)]]
make_graphs('hex', params)
make_graphs('bump', params)
make_graphs('kink', params)

params[0][2] = params[1][2] = 1.35
make_graphs('hexbig', params)