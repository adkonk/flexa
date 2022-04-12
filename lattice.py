#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from FlexaSheet import FlexaSheet

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

def hex_with_noise(noise=0.1):
    n_steps = 11 # how many flexas to put in interval [-10, 10]
    
    line = np.linspace(-10, 10, n_steps)
    dx = line[1] - line[0]
    x, y = np.meshgrid(line, np.sqrt(3) * line / 2)
    x[::2, :] += (20 / (n_steps - 1)) / 2
    
    # lattice coordinates for cell positions
    lattice = np.stack((x.flatten(), y.flatten(), np.zeros(n_steps ** 2)), axis = 1)
    
    lattice[:, :2] += noise * dx * \
        (np.random.random_sample((lattice.shape[0], 2)) - 0.5)
    
    lattice = lattice[np.linalg.norm(lattice, axis = 1) < 5, :]
    return(lattice)

def random_lattice():
    n_points = 100
    lattice = np.random.random((n_points, 2))
    lattice = lattice[np.linalg.norm(lattice, axis=1) < 0.5, :]
    return(lattice)
    
#%% test script

if __name__ == '__main__':
    h = hex_lattice()
    f = FlexaSheet(h, 0.8, 0.8, constrained = True)
    f.solve_shape()
    
    plt.figure(figsize=(16,16))
    f.draw()
    
    plt.figure(figsize=(20,8))
    f.draw(style='3d')
    plt.show()