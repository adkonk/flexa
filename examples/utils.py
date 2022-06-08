import numpy as np
import matplotlib.pyplot as plt
import os
from inspect import stack

from flexa.lattice import hex_lattice, hex_with_kink, \
    hex_with_bump
from flexa.FlexaSheet import FlexaSheet
from flexa._utils import picklify

name_to_func = {'hex': hex_lattice, 'hexbig': lambda: hex_lattice(15),
                'kink': hex_with_kink, 'bump': hex_with_bump}

def save_folder_path(folder_name, make=True):
    f = stack()[-1].filename # filename of file save_folder_path called from
    parent_dir = os.path.dirname(os.path.realpath(f))
    path = os.path.join(parent_dir, folder_name)
    if make and not os.path.isdir(path):
        os.mkdir(path)
    return(path)

def nameroot(name, phi0, psi0, ell0, k):
    return('%s%0.2f_%0.2f_%0.2f_%d' % (name, phi0, psi0, ell0, k[2]))

def file_path(save_folder_path, name, phi0, psi0, ell0, k):
    fname = nameroot(name, phi0, psi0, ell0, k)
    fpath = os.path.join(save_folder_path, picklify(fname))
    return(fpath)

def spiral_inds(X):
    # returns indices to traverse X in spiral order
    # reverse the output of this function to start from center    
    shape = np.array(X.shape)
    inds = np.zeros((X.size, 2)).astype('int')

    visited = np.zeros(X.shape).astype('bool')

    pos = np.array([0, 0]) # current position
    dir = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]]) # possible directions
    dir_i = 0 # current direction
    for i in np.arange(X.size):
        inds[i, :] = pos
        visited[inds[i, 0], inds[i, 1]] = True

        new_pos = pos + dir[dir_i]
        if np.any(new_pos >= shape): # hit a wall
            dir_i = (dir_i + 1) % 4
        elif visited[new_pos[0], new_pos[1]]: # already encountered this index
            dir_i = (dir_i + 1) % 4
        pos += dir[dir_i]
    
    return(inds)

def meshplotter(x, y, data, path_func, title='', cbarlabel='', 
        log=False, vmin=None, vmax=None):
    x_avgs = (x[:-1] + x[1:]) / 2
    x_bounds = [0, *x_avgs, x[-1] + 1]

    y_avgs = (y[:-1] + y[1:]) / 2
    y_bounds = [0, *y_avgs, y[-1] + 1]

    xbounds, ybounds = np.meshgrid(x_bounds, y_bounds)

    fig = plt.figure(figsize=(20,10))
    grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)

    ax = fig.add_subplot(grid[:,:2])

    def lognone(x):
        if x is None:
            return(None)
        else:
            return(np.log10(x))
    if log:
        data = np.log10(data)
    
    mesh = plt.pcolormesh(xbounds, ybounds, data, 
        vmin=lognone(vmin), vmax=lognone(vmax))

    plt.title(title, fontsize=20)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    plt.xlim([x[0] - dx / 2, x[-1] + dx / 2])
    plt.ylim([y[0] - dy / 2, y[-1] + dy / 2])

    plt.xlabel(r'$\phi$', fontsize=14)
    plt.ylabel(r'$\psi$', fontsize=14)

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    if cbarlabel == '':
        cbarlabel = title

    if log:
        cbarlabel = r'$\log_{10}$(' + cbarlabel + ')'
    cbar.set_label(cbarlabel, rotation=270, fontsize=14)

    X, Y = np.meshgrid(x, y)
    plt.plot(X, Y, 'k.', markersize=1)
    
    pairs = np.array([[0, -1], [-1, -1], [0, 0], [-1, 0]])
    plt.plot(x[pairs[:,0]], y[pairs[:,1]], 'wx', ms=13)

    for (pos, inds) in zip([[0,2],[0,3],[1,2],[1,3]], pairs):
        ax = fig.add_subplot(grid[pos[0], pos[1]], projection='3d')
        # ax.set_axis_off()
        fpath = path_func(x[inds[0]], y[inds[1]])
        s = FlexaSheet.load(fpath, silent=1)
        s.draw('3d', ax=ax)

        plt.title(r'$\phi = %0.2f$, $\psi = %0.2f$' %
                    (x[inds[0]], y[inds[1]]))

    return mesh