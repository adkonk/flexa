#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from FlexaSheet import FlexaSheet
from icosphere import icosphere 

#%% start functions

def hex_lattice():
    n_steps = 11 # how many flexas to put in interval [-10, 10]
    
    line = np.linspace(-10, 10, n_steps)
    x, y = np.meshgrid(line, np.sqrt(3) * line / 2)
    x[::2, :] += (20 / (n_steps - 1)) / 2
    
    # lattice coordinates for cell positions
    lattice = np.stack((x.flatten(), y.flatten(), np.zeros(n_steps ** 2)), 
                       axis = 1)
    
    lattice = lattice[np.linalg.norm(lattice, axis = 1) < 5, :]
    return(lattice)

def hexbig_lattice():
    n_steps = 15 # how many flexas to put in interval [-10, 10]
    
    line = np.linspace(-10, 10, n_steps)
    x, y = np.meshgrid(line, np.sqrt(3) * line / 2)
    x[::2, :] += (20 / (n_steps - 1)) / 2
    
    # lattice coordinates for cell positions
    lattice = np.stack((x.flatten(), y.flatten(), np.zeros(n_steps ** 2)), 
                       axis = 1)
    
    lattice = lattice[np.linalg.norm(lattice, axis = 1) < 5, :]
    return(lattice)

def hex_with_noise(noise=0.3):
    lattice = hex_lattice()
    dx = np.linalg.norm(lattice[0, :] - lattice[1, :])
    
    lattice[:, :2] += noise * dx * \
        (np.random.random_sample((lattice.shape[0], 2)) - 0.5)
    return(lattice)

def hex_with_kink():
    lattice = hex_lattice()
    lattice = lattice[np.linalg.norm(lattice - np.array([0, 1, 0]), axis=1) > 2, :]
    
    lattice = np.concatenate((lattice, np.array([[0, 1.3, 0]])), axis=0)
    return(lattice)

def hex_with_bump():
    lattice = hex_lattice()
    
    lattice = np.concatenate((lattice, np.array([[0, 0.8, 0]])), axis=0)
    return(lattice)

def random_lattice(n):
    lattice = np.random.random((n, 2))
    lattice = lattice[np.linalg.norm(lattice, axis=1) < 0.5, :]
    return(lattice)
    
def ico(ndiv, radius=5, angle=np.pi/3, lift=False):
    v, f = icosphere(ndiv)
    
    # trim v
    keep_v = np.arccos(-v[:, 2] / np.linalg.norm(v, axis=1)) <= angle
    v = v[keep_v, :]
    v *= radius
    if lift: v[:, 2] += radius

    # trim f
    keep_f = np.all(np.isin(f, np.where(keep_v)[0]), axis=1)
    f = f[keep_f, :]

    f = rankdata(f, method='dense').reshape((-1, 3)) - 1

    return((v, f))

def random_sphere_points(n, radius=5, angle=np.pi/3):
    zmax = np.cos(angle)
    z = np.random.uniform(-1, zmax, size=n)

    thetas = np.arccos(z)
    phis = np.random.uniform(0, 2 * np.pi, size=n)

    x = np.column_stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)))
    return(radius * x)

#%% test script

if __name__ == '__main__':
    
    #% flipping a big sheet
    h = hexbig_lattice()#hex_lattice_with_kink() # hex_with_noise()
    
    f = FlexaSheet.flatgen(h, 0.8, 0.8, 1.349, constrained = False)
    plt.figure(figsize=(10,10))
    f.draw()
    plt.savefig('hexbig_graph.png', dpi=200)
    
    f.solve_shape(10)
#    f.plot_energies()
#    plt.savefig('hex0.8_0.8_1.52_10_energies.png', dpi=200)
    
    plt.figure(figsize=(16,16))
    f.draw()
    plt.savefig('hexbig0.8_0.8_1.35_10_graph.png', dpi=200)
    
    plt.figure(figsize=(20,8))
    f.draw(style='3d')
    plt.savefig('hexbig0.8_0.8_1.35_10_plot.png', dpi=200)
    
    
    f.phi0 = 0.95
    f.solve_shape(10)
#    f.plot_energies()
#    plt.savefig('hex0.95_0.8_1.52_10_energies.png', dpi=200)
    
    plt.figure(figsize=(16,16))
    f.draw()
    plt.savefig('hexbig0.95_0.8_1.35_10_graph.png', dpi=200)
    
    plt.figure(figsize=(20,8))
    f.draw(style='3d')
    plt.savefig('hexbig0.95_0.8_1.35_10_plot.png', dpi=200)

    #% simulating an icosahedron shape
    v, f = lattice.ico(3, angle=2*np.pi/5)

    s = FlexaSheet.facegen(v, f, phi0=0.654, psi0=0.8375, ell0=1)
    s.draw('3d')
    plt.show()
    
    
    