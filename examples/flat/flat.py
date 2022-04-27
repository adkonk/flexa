
from flexa.lattice import hex_lattice, hexbig_lattice, hex_with_kink, \
    hex_with_bump
from collections.abc import Iterable

import os

# used to save figures in the same directory as this script
dir_path = os.path.dirname(os.path.realpath(__file__))

name_to_func = {'hex': hex_lattice, 'hexbig': hexbig_lattice,
                'kink': hex_with_kink, 'bump': hex_with_bump}

def nameroot(name, phi0, psi0, ell0, k):
    return('%s%0.2f_%0.2f_%0.2f_%d' % (name, phi0, psi0, ell0, k))

def make_graphs(name, params):
    # params takes form [(phi0, psi0, ell0, k)] 
    # list of lists of parameters
    f = name_to_func[name]
    lattice = f()

    # make sure params is list of lists
    if not isinstance(params[0], Iterable):
        params = [params]

    f = FlexaSheet.flatgen(lattice, params[0][0], params[0][1], params[0][2])
    plt.figure(figsize=(10,10))
    f.draw()
    plt.savefig('%s_graph.png' % name, dpi=200)

    for (phi0, psi0, ell0, k) in params:
        f.phi0 = phi0 
        f.psi0 = psi0
        f.ell0 = ell0
        f.solve_shape(k)
        
        r = nameroot(name, phi0, psi0, ell0, k)
        
        plt.figure(figsize=(16,16))
        f.draw()
        plt.savefig(os.path.join(dir_path, '%s_graph.png' % r), dpi=200)
        
        plt.figure(figsize=(20,8))
        f.draw(style='3d')
        plt.savefig(os.path.join(dir_path, '%s_plot.png' % r), dpi=200)

params = np.array([[0.8, 0.8, 1.52, 10], [0.95, 0.8, 1.52, 10]])
make_graphs('hex', params)
make_graphs('bump', params)
make_graphs('kink', params)

params[:, 2] = 1.35
make_graphs('hexbig', params)