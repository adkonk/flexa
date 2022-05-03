import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d
from scipy.optimize import minimize
import time
import pickle

class FlexaSheet(object):
	def __init__(self, G, phi0=None, psi0=None, ell0=None,
				 constrained=False, silent=0):
		""" Generates the FlexaSheet object
		Parameters:
			G: networkx graph
		"""
		
		self.G = G

		pos = nx.get_node_attributes(G, 'x0')
		self.x = np.array(list(pos.values())).flatten()
		r = self.x.reshape((-1, 3))
		self.n = r.shape[0]

		# number of cells
		self.n_cells = sum(nx.get_node_attributes(self.G, 'cell').values())

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
		
		if silent == 0:
			print('\nInitial energies')
			self.energy_stats(r)
			print('\nInitial geometry:')
			self.geom_stats(r)

	@classmethod
	def flatgen(cls, x0, phi0=None, psi0=None, ell0=None, constrained=False,
			silent=0):
		assert x0.shape[1] == 2 or (x0.shape[1] == 3 and np.all(x0[:, 2] == 0))
		
		if x0.shape[1] == 2:
			x0 = np.concatenate((x0, np.zeros((x0.shape[0], 1))), axis=1)
		
		n_cells = x0.shape[0]

		vor = Voronoi(x0[:, :2])
		collars = np.concatenate((vor.vertices, 
									np.ones((vor.vertices.shape[0], 1))),
						axis=1)
		
		G = nx.Graph()
		# add cell vertices
		G.add_nodes_from([(i, {'x0': x0[i, :], 'cell': True}) 
			for i in range(n_cells)])
	
		# add collar vertices
		G.add_nodes_from(
				[(i + n_cells, {'x0': collars[i, :], 'cell': False}) 
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
				G.add_edge(cells[0], verts[0] + n_cells, collar=True)
				G.add_edge(cells[1], verts[0] + n_cells, collar=True)
				
				G.add_edge(cells[0], verts[1] + n_cells, collar=True)
				G.add_edge(cells[1], verts[1] + n_cells, collar=True)
					
				G.add_edge(cells[0], cells[1], collar=False, 
							collar_pts=[verts[0] + n_cells, 
										verts[1] + n_cells])
			if verts[0] == -1:
				bad_edges1.append([cells, verts])
				new_nodes += 1
			if verts[1] == -1:
				bad_edges2.append([cells, verts])
				new_nodes += 1
		
		# place collars at center of mass of their cells
		# before we add collars on the outside of the sheet
		cell_booleans = nx.get_node_attributes(G, 'cell')
		for (i, b) in cell_booleans.items():
			if not b:
				pos[i] = np.mean([pos[j] for j in G[i]], axis=0) + \
					np.array([0, 0, 1])
		nx.set_node_attributes(G, pos, 'x0')
		
		for (cells, verts) in bad_edges1:
			# make a new collar node equally distant as the existing collar 
			# node
			# find vector to existing collar node
			dirr = pos[verts[1] + n_cells] - pos[cells[0]]
			
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
						collar_pts=[verts[1] + n_cells, new_node])
			
		for (cells, verts) in bad_edges2:
			dirr = pos[verts[0] + n_cells] - pos[cells[0]]
		
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
						collar_pts=[verts[0] + n_cells, new_node])

		return(cls(G, phi0, psi0, ell0, constrained, silent))

	@classmethod
	def facegen(cls, x0, faces, phi0=None, psi0=None, ell0=None,
				constrained=False, silent=0):
		G = FlexaSheet.cellgraph(x0)

		neigh_collars = dict()
		def sortkey(i, j):
			assert i != j
			return(min(i, j), max(i, j))

		for (i, j) in zip(faces[:, 0], faces[:, 1]):
			neigh_collars[sortkey(i, j)] = []
		for (i, j) in zip(faces[:, 1], faces[:, 2]):
			neigh_collars[sortkey(i, j)] = []
		for (i, j) in zip(faces[:, 2], faces[:, 0]):
			neigh_collars[sortkey(i, j)] = []
		
		if np.all(np.equal([len(f) for f in faces], 3)):
			normfunc = FlexaSheet.tri_normal
		else: 
			normfunc = FlexaSheet.face_normal

		if ell0 is None:
			ell0 = 1

		edge_perps = dict()

		for f in faces:
			# TODO: add argument to make ref point towards the origin
			# for every face
			ref = np.array([0, 0, 1])
			n = normfunc(x0[f, :], ref=ref)
			x = np.mean(x0[f, :], axis=0) + ell0 * n

			edge_perps[sortkey(f[0], f[1])] = n
			edge_perps[sortkey(f[1], f[2])] = n
			edge_perps[sortkey(f[2], f[0])] = n

			new_node = G.number_of_nodes()
			G.add_node(new_node, x0=x, cell=False)

			G.add_edge(f[0], new_node, collar=True)
			G.add_edge(f[1], new_node, collar=True)
			G.add_edge(f[2], new_node, collar=True)

			neigh_collars[sortkey(f[0], f[1])].append(new_node)
			neigh_collars[sortkey(f[1], f[2])].append(new_node)
			neigh_collars[sortkey(f[2], f[0])].append(new_node)

		pos = nx.get_node_attributes(G, 'x0')

		for (k, v) in neigh_collars.items():
			assert len(v) > 0 and len(v) <= 2
			i = k[0]
			j = k[1]
			if len(v) == 1:
				new_node = G.number_of_nodes()
				x = pos[v[0]] # existing collar position

				n = edge_perps[k] # normal of the face edge k is part of 
				ab = pos[i] - pos[j] # vector of edge k
				m = np.cross(ab, n) # normal vector of plane given by ab, n
				k = np.dot(m, pos[i]) # offset when ab, n plane is given 
									  # m dot y = k

				c = (k - np.dot(m, x)) / np.dot(m, m) # closest point on ab, n
													  # plane to x is given
													  # by x + cm

				G.add_node(new_node, x0=(x + 2 * c * m), cell=False)
				G.add_edge(i, new_node, collar=True)
				G.add_edge(j, new_node, collar=True)
				v.append(new_node)

			G.add_edge(i, j, collar=False, collar_pts=v)

		return(cls(G, phi0, psi0, ell0, constrained, silent))

	@classmethod
	def trisurfgen(cls, x0, phi0=None, psi0=None, ell0=None, constrained=False):
		G = FlexaSheet.cellgraph(x0)
		tri = Delaunay(x0)

		# TODO: implement
		raise NotImplementedError('not done yet')

	@staticmethod
	def cellgraph(x0):
		G = nx.Graph()
		# add cell vertices
		G.add_nodes_from([(i, {'x0': x0[i, :], 'cell': True}) 
			for i in range(x0.shape[0])])
		return(G)

	@staticmethod
	def collar_pairs(neigh_collars):
		collar_pairs = zip(list(neigh_collars.values()))
		collar_pairs = np.array(list(collar_pairs)).flatten().reshape(-1, 2)
		return(collar_pairs)
	
	def energy_stats(self, r):
		e_phi = self.phi_energy(r)
		e_psi = self.psi_energy(r)
		e_spring = self.spring_energy(r)
		print(
			  'phi energy: %0.3e\n' % e_phi + \
			  'psi energy: %0.3e\n' % e_psi + \
			  'spring energy: %0.3e' % e_spring
			  )
		return(e_phi, e_psi, e_spring)
	
	def geom_stats(self, r):
		print(
			  'Average phi: %0.3e\n' % self.aphi(r) + \
			  'Average psi: %0.3e\n' % self.apsi(r) + \
			  'Average collar length: %0.3e\n' % np.mean(self.collar_lengths(r))
			  )
	
	## network
	def cell_degrees(self):
		return({c: len(collars) for (c, collars) in self.cell_collars.items()})
	
	## simulation
	# basic algebra
	@staticmethod
	def angle(a, b):
		return(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))
	
	@staticmethod 
	def tri_normal(a, b=None, c=None, ref=np.array([0, 0, 1])):
		# TODO: check inputs better
		if (len(a.shape) == 1 and a.size == 2) \
			or (len(a.shape) == 2 and a.shape[1] == 2): 
			return ref 

		if b is None and c is None:
			assert len(a.shape) == 2 and a.shape[0] == 3
			b = a[1, :]
			c = a[2, :]
			a = a[0, :]

		n = np.cross(b - a, c - a)
		n = n / np.linalg.norm(n)
		if np.dot(n, ref) < 0:
			n *= -1
		return(n)
	
	@staticmethod
	def face_normal(r, ref=np.array([0, 0, 1])):
		# solve least squares approximation z_i = (x_i, y_i) * (v_1, v_2) + c
		# so then the normal vector for plane is (v_1, v_2, -1)
		x = np.concatenate((r[:, :2], np.ones((r.shape[0], 1))), 
						   axis = 1)
		v = np.linalg.inv(x.T @ x) @ x.T @ r[:, [2]]
		v = np.array([v[0, 0], v[1, 0], -1])
	
		v /= np.linalg.norm(v)

		if np.dot(v, ref) < 0:
			return(-1 * v)
		else:
			return(v)

	def cell_normal(self, r, cell):
		rcis = r[self.cell_collars[cell], :]
		rc = r[cell, :]
		if len(self.cell_collars[cell]) == 3:
			return(FlexaSheet.tri_normal(rcis, ref=np.sum(rcis - rc, axis=0)))
		else:
			return(FlexaSheet.face_normal(rcis, np.sum(rcis - rc, axis=0)))
	
	# calculating the energies
	def phis(self, r):
		ps = {}
		for (c, collar_nodes) in self.cell_collars.items():
			n = self.cell_normal(r, c)
			ps[c] = np.array([FlexaSheet.angle(n, r[ci,:] - r[c,:]) \
			  for ci in collar_nodes])
		return(ps)
	
	# average phi
	def aphi(self, r):
		ps = self.phis(r)
		phi_tot = np.sum([p for cell_phis in ps.values() for p in cell_phis])
		n = sum(self.cell_degrees().values())
		return(phi_tot / n)
	
	def phi_energies(self, r):
		return({i: np.sum((phis_c - self.phi0) ** 2) \
				for (i, phis_c) in self.phis(r).items()})
	
	def phi_energy(self, r):
		e = sum(self.phi_energies(r).values())
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
	
	def psi_energies(self, r):
		return({uv: (psi - self.psi0) ** 2 \
				for (uv, psi) in self.psis(r).items()})
	
	def psi_energy(self, r):
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
	def solve_shape(self, k_spring=0, silent=0):
		# silent 0: complete info, 1: time elapsed only, 2: nothing
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
		if silent == 0:
			print('\nBeginning solver')
		t = time.time()
		res = minimize(
				self.energy, self.x, method='SLSQP', 
				args = (k_spring),
				constraints = eq_cons
				)
		if silent in [0, 1]:
			print('solver complete\nTime elapsed : %0.2f minutes' % \
				((time.time() - t) / 60))
		# return the entire optimiser result so we keep all information
		self.x = res.x
		if silent == 0:
			print('\nFinal energies')
			self.energy_stats(self.x.reshape((-1, 3)))
		return(res)
	 
	## shape analysis
	def plot_energies(self, ):
		# TODO: develop this
		r = self.x.reshape((-1, 3))
		
		fig = plt.gcf()
		# phi energies
		ax1 = fig.add_subplot(2, 1, 1)
		ax1.plot(np.linalg.norm(r[:self.n_cells, :], axis = 1),
			self.phi_energies(r).values(),
			'.')
	
		# psi energies
		ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
		collar_pairs = FlexaSheet.collar_pairs(self.neigh_collars)
		ax2.plot(
			np.linalg.norm((r[collar_pairs[:, 0], :] + \
						   r[collar_pairs[:, 1], :]) / 2, axis=1),
			self.psi_energies(r).values(),
			'.')
	
	# TODO: add statistics for cells of different degrees
	
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
			
			ax_given = True
			if ax is None:
				ax = fig.add_subplot(1, 2, 1, projection='3d')
				ax_given = False
			# TODO: add Triangulation argument to specify cell sheet topology
			ax.plot_trisurf(r[:self.n_cells, 0], 
							r[:self.n_cells, 1], 
							r[:self.n_cells, 2],
							cmap='spring', edgecolor='black')
			ax.scatter3D(r[self.n_cells:, 0], 
						 r[self.n_cells:, 1], 
						 r[self.n_cells:, 2])
			xflines = r.reshape(-1, 1, 3) # format for plotting lines, idk why
			collar_pairs = FlexaSheet.collar_pairs(self.neigh_collars)
			# TODO: add colors based on distance from camera
			c = Line3DCollection(
				np.concatenate([xflines[collar_pairs[:, 1]], 
								xflines[collar_pairs[:, 0]]], 
							   axis = 1)
				)
			ax.add_collection3d(c)
			if ax_given:
				return(ax)
			
			ax.view_init(10, 20)
			# ax.set_zlim(-1, 2) 
			
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
			# ax.set_zlim(-1, 2) 
			
			plt.tight_layout()
		# TODO: add plotter showing phi energy at each cell and 
		# psi energy at each boundary
	
	## file management
	@staticmethod
	def picklify(name):
		ext = '.p'
		if not name.lower().endswith(ext):
			name = name + ext
		return(name)

	def __eq__(self, other):
		if isinstance(other, FlexaSheet):
			return np.all(self.x == other.x) and \
				self.cell_collars == other.cell_collars and \
				self.neigh_collars == other.neigh_collars
		return(False)

	def save(self, name):
		data = {}

		data['init_params'] = {'G': self.G, 'phi0': self.phi0, 
			'psi0': self.psi0, 'ell0': self.ell0, 
			'constrained': self.constrained}
		
		data['attributes'] = {'x': self.x}

		with open(FlexaSheet.picklify(name), 'wb') as f:
			pickle.dump(data, f)

	@classmethod
	def load(cls, name, silent=1):
		with open(FlexaSheet.picklify(name), 'rb') as f:
			saved = pickle.load(f)
		
		data = saved['init_params']
		s = cls(data['G'], data['phi0'], data['psi0'], data['ell0'],
			data['constrained'], silent)
		
		s.__dict__.update(saved['attributes'])
		
		return(s)
