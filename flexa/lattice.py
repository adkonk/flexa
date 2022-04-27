import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from FlexaSheet import FlexaSheet
from icosphere import icosphere 

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
