# -*- coding: utf-8 -*-
#%% imports

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import minimize
import time

#%% class definition

class FlexaSheet(object):
    def __init__(self, x0: np.ndarray, phi0=None, psi0=None, ell0=None,
                 constrained=False):
        """ Generates the FlexaSheet object
        Parameters:
            x0: initial (n, 2) or (n, 3) array of cell coordinates
        """
        self.n_cells = x0.shape[0]
        
        print('Creating a sheet with: \n%d cells' % self.n_cells)
        
        if x0.shape[1] == 2 or (x0.shape[1] == 3 and np.all(x0[:, 2] == 0)):
            if x0.shape[1] == 2:
                x0 = np.concatenate((x0, np.zeros((x0.shape[0], 1))), axis=1)
            
            vor = Voronoi(x0[:, :2])
            collars = np.concatenate((vor.vertices, 
                                      np.ones((vor.vertices.shape[0], 1))),
                         axis=1)
            
            G = nx.Graph()
            # add cell vertices
            G.add_nodes_from([(i, {'x0': x0[i, :], 'cell': True}) 
                for i in range(self.n_cells)])
        
            # add collar vertices
            G.add_nodes_from(
                    [(i + self.n_cells, {'x0': collars[i, :], 'cell': False}) 
                     for i in range(collars.shape[0])])  
    
            pos = nx.get_node_attributes(G, 'x0')

            bad_edges1 = []
            bad_edges2 = []
            new_nodes = 0
            for i in range(len(vor.ridge_points)):
                cells = vor.ridge_points[i, :]
                verts = vor.ridge_vertices[i]
                  
                # make cell-collar edges
                if verts[0] != -1 and verts[1] != -1:
                    G.add_edge(cells[0], verts[0] + self.n_cells, collar=True)
                    G.add_edge(cells[1], verts[0] + self.n_cells, collar=True)
                    
                    G.add_edge(cells[0], verts[1] + self.n_cells, collar=True)
                    G.add_edge(cells[1], verts[1] + self.n_cells, collar=True)
                      
                    G.add_edge(cells[0], cells[1], collar=False, 
                               collar_pts=[verts[0] + self.n_cells, 
                                           verts[1] + self.n_cells])
                if verts[0] == -1:
                    bad_edges1.append([cells, verts])
                    new_nodes += 1
                if verts[1] == -1:
                    bad_edges2.append([cells, verts])
                    new_nodes += 1
                
            for (cells, verts) in bad_edges1:
                # make a new collar node equally distant as the existing collar 
                # node
                # find vector to existing collar node
                dirr = pos[verts[1] + self.n_cells] - pos[cells[0]]
                
                # find vector between cells
                center_line = pos[cells[1]] - pos[cells[0]]
                # find orthogonal component of dir on center_line
                dirr -= center_line * np.dot(dirr, center_line) / \
                    np.dot(center_line, center_line)
                dirr *= np.array([1, 1, -1])
                
                new_node = G.number_of_nodes()
                G.add_node(new_node, x0=(pos[cells[0]] + 
                                         pos[cells[1]]) / 2 - dirr, 
                           cell=False)
                G.add_edge(cells[0], new_node, collar=True)
                G.add_edge(cells[1], new_node, collar=True)
                
                G.add_edge(cells[0], cells[1], collar=False,
                           collar_pts=[verts[1] + self.n_cells, new_node])
              
            for (cells, verts) in bad_edges2:
                dirr = pos[verts[0] + self.n_cells] - pos[cells[0]]
            
                center_line = pos[cells[1]] - pos[cells[0]]
                dirr -= center_line * np.dot(dirr, center_line) / \
                    np.dot(center_line, center_line)
                dirr *= np.array([1, 1, -1])

                new_node = G.number_of_nodes()
                G.add_node(new_node, x0=(pos[cells[0]] + 
                                         pos[cells[1]]) / 2 + dirr, 
                           cell=False)
                G.add_edge(cells[0], new_node, collar=True)
                G.add_edge(cells[1], new_node, collar=True)

                G.add_edge(cells[0], cells[1], collar=False,
                           collar_pts=[verts[0] + self.n_cells, new_node])
                
            self.G = G
            
            # dict {cell: [cell collar nodes]}
            self.cell_collars = {}
            for i in range(self.n_cells):
                self.cell_collars[i] = [k for (k, v) in G[i].items() \
                                 if v['collar']]
             
            # dict {(cell1, cell2): (collar node 1, collar node 2)}
            self.neigh_collars = nx.get_edge_attributes(G, 'collar_pts')
            
            # list of (cell, collar node) or (collar node, cell)
            self.collar_edges = np.array(
                [(e[0], e[1]) for e in self.G.edges.data('collar') if e[2]])
            
            # TODO: fix the startup message
            print('%d cell-cell interactions \n' % len(G.edges) + \
                  '%d boundary collar nodes' % len(G.edges))
            
        elif x0.shape[1] == 3:
            raise NotImplementedError('3d starting coordinates ' + \
                  'not yet implemented!')
        else:
            raise NotImplementedError('?')
        
        pos = nx.get_node_attributes(G, 'x0')
        self.x = np.array(list(pos.values())).flatten()
        r = self.x.reshape((-1, 3))
        
        self.n = r.shape[0]
        
        if phi0 is None:
            self.phi0 = self.aphi(r)
        else: self.phi0 = phi0
        if psi0 is None:
            self.psi0 = self.apsi(r)
        else: self.psi0 = psi0
        self.constrained = constrained
        if ell0 is None:
            self.ell0 = self.collar_lengths(r)
        else:
            if self.constrained:
                assert np.all(self.collar_lengths(r) == ell0)
            self.ell0 = ell0
        
        print(
              'Initial phi energy: %0.3f\n' % self.phi_energy(r) + \
              'Initial psi energy: %0.3f\n' % self.psi_energy(r) + \
              'Initial spring energy: %0.3f' % self.spring_energy(r)
              )
    
    ## network
    def cell_degrees(self):
        return({c: len(collars) for (c, collars) in self.cell_collars.items()})
    
    ## simulation
    # basic algebra
    @staticmethod
    def angle(a, b):
        return(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))
    
    def cell_normal(self, r, cell):
        rcis = r[self.cell_collars[cell], :]
        rc = r[cell, :]
        
        # solve least squares approximation z_i = (x_i, y_i) * (v_1, v_2) + c
        # so then the normal vector for plane is (v_1, v_2, -1)
        x = np.concatenate((rcis[:, :2], np.ones((rcis.shape[0], 1))), 
                           axis = 1)
        v = np.linalg.inv(x.T @ x) @ x.T @ rcis[:, [2]]
        v = np.array([v[0, 0], v[1, 0], -1])
    
        # but we don't know the orientation of the normal vector.
        # here we could use the average cell-to-collar vector to fix the 
        # orientation
        to_collar = np.sum(rcis - rc, axis=0)
        if np.dot(v, to_collar) < 0:
          return(-1 * v)
        else:
          return(v)
    
    # calculating the energies
    def phis(self, r):
        ps = {}
        for (c, collar_nodes) in self.cell_collars.items():
            n = self.cell_normal(r, c)
            ps[c] = [FlexaSheet.angle(n, r[ci,:] - r[c,:]) \
                for ci in collar_nodes]
        return(ps)
    
    # average phi
    def aphi(self, r):
        ps = self.phis(r)
        phi_tot = np.sum([p for cell_phis in ps.values() for p in cell_phis])
        n = sum(self.cell_degrees().values())
        return(phi_tot / n)
        
    def phi_energy(self, r):
        e = sum([(phi - self.phi0) ** 2 for phis_c in self.phis(r).values() \
                 for phi in phis_c])
        return(e)
    
    def psis(self, r):
        ps = {}
        for (cells, collars) in self.neigh_collars.items():
            # find normal vector a for cell 1 to shared collar boundary
            c11 = r[collars[0], :] - r[cells[0], :]
            c12 = r[collars[1], :] - r[cells[0], :]
            a = np.cross(c11, c12)
            n1 = np.sum(r[self.cell_collars[cells[0]], :] - r[cells[0], :], 
                        axis=0)
            # flip a if it doesn't line up with average cell to collar vec n1
            if np.dot(n1, a) < 0:
                a *= -1
    
            # find normal vector b for cell 2 to shared collar boundary
            c21 = r[collars[0], :] - r[cells[1], :]
            c22 = r[collars[1], :] - r[cells[1], :]
            n2 = np.sum(r[self.cell_collars[cells[1]], :] - r[cells[1], :], 
                        axis=0)
            b = np.cross(c21, c22)
            if np.dot(n2, b) < 0:
                b *= -1
    
            # get the angle on the inside of the hinge between the two cells
            # at the collar boundary 
            psi = np.pi - FlexaSheet.angle(a, b)
            ps[cells] = psi / 2
        return(ps)
    
    def apsi(self, r):
        ps = self.psis(r)
        psi_tot = sum([p for p in ps.values()])
        n = len(self.neigh_collars)
        return(psi_tot / n)
    
    def psi_energy(self, r):
        '''
        Calculate the collar-collar interaction energy
        '''
        e = sum([(psi - self.psi0) ** 2 for psi in self.psis(r).values()])
        return(e)

    def collar_lengths(self, r):
        return(np.linalg.norm(r[self.collar_edges[:, 0], :] - \
                              r[self.collar_edges[:, 1], :], axis = 1))

    def spring_energy(self, r):
        return(np.sum((self.collar_lengths(r) - self.ell0) ** 2))

    def energy(self, x, k_spring=0):
      r = x.reshape((-1, 3))
      e = self.phi_energy(r) + self.psi_energy(r)
      e += k_spring * self.spring_energy(r) 
      return(e)   
     
    # simulate
    def solve_shape(self, k_spring=0):
        n_edges = self.collar_edges.shape[0]
         
        # Jacobian matrix of the above function
        def J_collar_lengths(x):
            r = x.reshape((-1, 3))
            J = np.zeros((n_edges, self.n, 3))
            J[np.arange(n_edges), self.collar_edges[:, 0], :] = \
                2 * (r[self.collar_edges[:, 0], :] - \
                     r[self.collar_edges[:, 1], :])
            J[np.arange(n_edges), self.collar_edges[:, 1], :] = \
                -1 * J[np.arange(n_edges), self.collar_edges[:, 0], :]
            return(J.reshape((n_edges, 3 * self.n)))
            
        # defining the constraint in scipy format
        if self.constrained:
            eq_cons = {'type': 'eq',
                       'fun': lambda x:
                           self.collar_lengths(x.reshape((-1, 3))) - self.ell0,
                       'jac': lambda x: J_collar_lengths(x)}
            assert k_spring == 0
        else: eq_cons = []
        # blasting it with the minimisation routine
        print('Beginning solver')
        t = time.time()
        res = minimize(
                self.energy, self.x, method='SLSQP', 
                args = (k_spring),
                constraints = eq_cons
                )
        print('solver complete\nTime elapsed : %0.2f minutes' % \
              ((time.time() - t) / 60))
        # return the entire optimiser result so we keep all information
        self.x = res.x
        return(res)
      
    ## plotting
    def draw(self, style='flat', x0=False, nodes=[], edges=[], ax=None):
        if style == 'flat':
            if x0:
                pos = nx.get_node_attributes(self.G, 'x0')
            else:
                xf = self.x.reshape((-1, 3))
                pos = dict(zip(np.arange(xf.shape[0]), xf))
            pos = {i[0]: i[1][:2] for i in pos.items()}
            
            cell = nx.get_node_attributes(self.G, 'cell')
            collar_checks = nx.get_edge_attributes(self.G, 'collar')
            
            if ax is None:
                ax = plt.gca()
            nx.draw_networkx(self.G, pos=pos, with_labels=False,
                             node_color=[not c for c in cell.values()],
                             node_size=[160 * c + 10 for c in cell.values()],
                             edge_color=[not c for c in \
                                         collar_checks.values()],
                             edge_cmap=plt.get_cmap('copper'),
                             ax=ax)
            for n in nodes:
                plt.plot(pos[n][0], pos[n][1], '.g', ms=20)
            for e in edges:
                plt.plot([pos[e[0]][0], pos[e[1]][0]],
                         [pos[e[0]][1], pos[e[1]][1]],
                         '-r', lw=5)
            plt.axis('equal')
            
        if style == '3d':
            fig = plt.gcf()
            r = self.x.reshape((-1, 3))
            
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot_trisurf(r[:self.n_cells, 0], 
                            r[:self.n_cells, 1], 
                            r[:self.n_cells, 2],
                            cmap='spring', edgecolor='black')
            ax.scatter3D(r[self.n_cells:, 0], 
                         r[self.n_cells:, 1], 
                         r[self.n_cells:, 2])
            
            xflines = r.reshape(-1, 1, 3) # format for plotting lines, idk why
            collar_pairs = zip(list(self.neigh_collars.values()))
            collar_pairs = \
                np.array(list(collar_pairs)).flatten().reshape(-1, 2)
            
            c = Line3DCollection(
                np.concatenate([xflines[collar_pairs[:, 1]], 
                                xflines[collar_pairs[:, 0]]], 
                               axis = 1)
                )
            ax.add_collection3d(c)
            
            ax.view_init(10, 20)
            ax.set_zlim(-1, 2) 
            
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            c = Line3DCollection(
                np.concatenate([xflines[collar_pairs[:, 1]], 
                                xflines[collar_pairs[:, 0]]], 
                               axis = 1)
                )
            ax.add_collection3d(c)
            
            ax.plot_trisurf(r[:self.n_cells, 0], 
                            r[:self.n_cells, 1], 
                            r[:self.n_cells, 2],
                            cmap='spring', edgecolor='black')
            ax.scatter3D(r[self.n_cells:, 0], 
                         r[self.n_cells:, 1], 
                         r[self.n_cells:, 2])
            
            ax.view_init(40, 20)
            ax.set_zlim(-1, 2) 
            
            plt.tight_layout()

#%% test script
        
if __name__ == '__main__':
    print('test')
    