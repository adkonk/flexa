from statistics import NormalDist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial import Voronoi, Delaunay, SphericalVoronoi
from scipy.optimize import minimize
from scipy.stats import rankdata
import time
import pickle

from flexa._utils import *
from flexa._geom_utils import *

class FlexaSheet(object):
	"""A class to describe and simulate sheets of Choanaeca flexa sheets. 

	Attributes:
		G (nx.Graph): see __init__
		x (np.ndarray): flattened array of cell and collar coordinates in the 
			order given by G. The `(n, 3)` 2d array with position 3-vectors as
			rows are given by `x.reshape(-1, 3)`
		n (int): number of points (both cells and collars) with coordinates
		n_cells (int): number of cells in `G`
		cell_collars (dict): dictionary of `{cell: [collar nodes that cell
			is attached to]}`
		neigh_collars (dict): dictionary `{(cell 1, cell2): (collar node 1,
			collar node 2)}` where the collar nodes correspond to the 
			boundary between cells 1 and 2. In a sense, neigh_collars stores
			the dual graph information in the surface of the collar network
		collar_edges (list): list of edges between cells and collar nodes
		phi0 (float): equilibrium phi angle
		psi0 (float): equilibrium psi angle
		ell0 (float): equilibrium cell-collar length
		constrained (bool): are the cell-collar edge lengths fixed?
	"""
	def __init__(self, G, phi0=None, psi0=None, ell0=None,
				 constrained=False, silent=0):
		"""Create FlexaSheet object

		Args:
			G (nx.Graph): graph describing flexa sheet topology. Contains nodes
				with attributes `cell` (bool) and attributes `x0` (1d array).
				Has edges between cells and collar nodes with attributes 
				`collar=True` (bool) and edges between contacting cells
				with attributes `collar=False` (bool) and 
				`collar_pts=(c1, c2)` where `c1` and `c2` are the collar nodes
				corresponding to the boundary between the two cells.
			phi0 (float, optional): equilibrium phi angle. Defaults to average
				phi value in initial sheet.
			psi0 (float, optional): equilibrium psi angle. Defaults to average
				phi value in initial sheet.
			ell0 (float or np.ndarray, optional): equilibrium cell-collar 
				length, either constant for the whole sheet or given
				for each cell-collar edge. Defaults to initial lengths for 
				each cell-collar edge
			constrained (bool, optional): fixes cell-collar lengths. 
				Only used during simulation. Specified during initialisation 
				to ensure initial lengths are equal to ell0. Defaults to False.
			silent (int, optional): argument to silence initialisation. Can be 
				0: all messages printed, 1: no messages. Defaults to 0.
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
		
		if phi0 is None: # phi0 = average initial phi0
			self.phi0 = self.aphi(r)
		else: self.phi0 = phi0
		if psi0 is None:# psi0 = average initial psi0
			self.psi0 = self.apsi(r)
		else: self.psi0 = psi0
		self.constrained = constrained
		if ell0 is None: # ell0 = initial ell0s
			self.ell0 = self.collar_lengths(r)
		else:
			if self.constrained: # all initial lengths must == ell0 if constr
				assert np.all(self.collar_lengths(r) == ell0)
			self.ell0 = ell0
		# self.s0 = self.sector_angles(r)
		
		if silent == 0: # print stats
			print('\nInitial energies')
			self.energy_stats(r)
			print('\nInitial geometry:')
			self.geom_stats(r)

	@classmethod
	def vorgen(cls, r0, z=1, ref=np.array([0, 0, 1]), **kwargs):
		"""Generates a FlexaSheet object from an initial lattice of points 
		representing cell positions on the xy plane or on a sphere centered
		at the origin

		Args:
			r0 (np.ndarray): `(n_cells, 3)` array with cell coordinates
			z (float): collar offset
			**kwargs: passed to __init__

		Returns: 
			FlexaSheet: with cells at positions r0
		"""
		if np.all(r0[:, 2] == r0[0, 2]):
			flat = True
		else: flat = False # points on the sphere

		# TODO: remove collars because I set position to center of mass anyway
		if flat:
			vor = Voronoi(r0[:, [0, 1]])
			n_ridges = len(vor.ridge_vertices)
			n_verts = vor.vertices.shape[0]
			# dictionary (cell1, cell2): (verts1, verts2) (v to c)
			vtoc = {sortkey(tuple(vor.ridge_vertices[i])): \
				list(vor.ridge_points[i, :]) for i in range(n_ridges)}
			collars = np.concatenate( # collar node positions from Voronoi
				(vor.vertices, z * np.ones((n_verts, 1))), axis=1)
		else:
			radius = np.linalg.norm(r0[0, :])
			assert np.all(np.isclose(np.linalg.norm(r0, axis=1), radius)), \
				'points must either be on a sphere or close to it for vorgen'

			z_max = np.amax(r0[:, 2]) / radius

			# voronoi on points of radius 1 because sv gave issues with r >= 1
			vor = SphericalVoronoi(r0 / radius)
			vor.sort_vertices_of_regions()

			# reindex
			keep_verts = np.where(vor.vertices[:, 2] <= z_max + 0.1)[0]
			faces = reindex_list_of_lists(vor.regions, keep=keep_verts)
			
			# find points corresponding to ridge vertices
			vtoc = {}
			for face in faces:
				for (i, j) in pairs(face):
					if i != -1 or j != -1:
						vtoc[sortkey(i, j)] = []
			for (pt, face) in enumerate(faces):
				for (i, j) in pairs(face):
					if i != -1 or j != -1:
						vtoc[sortkey(i, j)].append(pt)
			
			collars = vor.vertices[keep_verts, :] * radius
			n_verts = collars.shape[0]

		n_cells = r0.shape[0]
		# create graph with cell vertices
		G = FlexaSheet.cellgraph(r0) 
	
		# add collar vertices
		G.add_nodes_from([(i + n_cells, {'x0': collars[i, :], 'cell': False}) \
			for i in range(n_verts)])  

		pos = nx.get_node_attributes(G, 'x0')

		# dict to help set bdary vert positions later
		bdary_collar_neighs = {}
		for (verts, cells) in vtoc.items():
			# make cell-collar edges
			if verts[0] != -1 and verts[1] != -1: # as long as bdary is finite
				for v in verts:
					for c in cells:
						G.add_edge(c, v + n_cells, collar=True)
				G.add_edge(cells[0], cells[1], collar=False, 
					collar_pts=[verts[0] + n_cells, verts[1] + n_cells])
			if verts[0] == -1:	
				new_node = G.number_of_nodes()
				# temp x0 to be changed later
				G.add_node(new_node, x0=np.array([0, 0, 0]), cell=False)
				for c in cells:
					G.add_edge(c, new_node, collar=True)
				
				G.add_edge(cells[0], cells[1], collar=False,
					collar_pts=[verts[1] + n_cells, new_node])
				bdary_collar_neighs[new_node] = verts[1] + n_cells
		
		cell_booleans = nx.get_node_attributes(G, 'cell')
		for (i, b) in [kv for kv in cell_booleans.items() if not kv[1]]:
			cells = [j for j in G[i] if cell_booleans[j]]
			if len(cells) > 2: # collar nodes on interior
				r = np.array([pos[j] for j in cells])
				n = face_normal(r, ref=ref)
				pos[i] = np.mean(r, axis=0) + z * n
			elif len(cells) == 2: # boundary collar nodes
				# find normal vector of the plane the neighboring collar's dual
				# TODO: this line occasionally throws KeyError, fix it
				plane_r = np.array([pos[j] for j in G[bdary_collar_neighs[i]]])
				n = face_normal(plane_r, ref=ref)

				# find closest point from neighboring collar to plane defined by
				# cell1, cell2, cell1 + n 
				closest = closest_point(pos[cells[0]], pos[cells[1]], 
					pos[cells[0]] + n, pos[bdary_collar_neighs[i]])

				# bdary collar position is reflection of neighbor collar through
				# c1, c2, c1 + n plane
				dirr = closest - pos[bdary_collar_neighs[i]]
				pos[i] = pos[bdary_collar_neighs[i]] + 2 * dirr
		nx.set_node_attributes(G, pos, 'x0')

		return(cls(G, **kwargs))

	@classmethod
	def flatgen(cls, r0, z=1, **kwargs):
		"""Generates a FlexaSheet object from an initial lattice of points 
		on the plane. Now just a wrapper for vorgen.

		Args:
			r0 (np.ndarray): `(n_cells, 2)` or `(n_cells, 3)` array giving 
				cell coordinates. If the latter, all z coordinates must be 0
			z (float): initial z position of collars. Defaults to 1
			**kwargs: passed to __init__
	
		Raises: 
			AssertionError: if r0 doesn't have 2 columns or has 3 columns but 
				the third isn't all zero

		Returns: 
			FlexaSheet: generated from r0, collars at z=1
		"""
		assert r0.shape[1] == 2 or (r0.shape[1] == 3 and np.all(r0[:, 2] == 0))
		
		if r0.shape[1] == 2: # add column of zeros to r0
			r0 = np.concatenate((r0, np.zeros((r0.shape[0], 1))), axis=1)
		return(FlexaSheet.vorgen(r0, z, ref=np.array([0, 0, 1]), **kwargs))

	@classmethod
	def facegen(cls, r0, faces, z=1, ref='ori', **kwargs):
		"""Generates FlexaSheet object from initial cell coordinates r0 and 
		faces (consisting of cell-cell interactions). 

		Creates initial collar positions based on the normals of the faces
		given in `faces` using either `tri_normal` or `face_normal` depending on
		the number of cells in each face. 

		Args:
			r0 (np.ndarray): `(n_cells, 3)` array giving cell coordinates.
			faces (list): list of lists containing indices for cells 
				corresponding to a face of the initial sheet. Assumed to be 
				cyclic as interactions between cells ([..., i, i+1, ...] 
				indicates that (i, i+1) is a cell-cell interaction)
			z (float): collar offset from cells in direction of face normals
			**kwargs: passed to __init__

		Raises: 
			AssertionError: if r0 does not have 3 columns

		Returns: 
			FlexaSheet: generated from r0 and faces
		"""
		assert r0.shape[1] == 3

		# create graph with cell vertices
		G = FlexaSheet.cellgraph(r0) 

		neigh_collars = dict() # {(cell1, cell2): (collar 1, collar 2)}

		# TODO: actual initial lengths are > z. Default ell0 to initial lengths?
		kwargs['ell0'] = z # equilibrium length will be set to z

		edge_perps = dict() # {(cell1, cell2): normal vec for one face with 
							# (cell1, cell2) as an edge}
		face_pos = dict()

		# add cell-cell edges as keys based on faces
		for f in faces:
			for (i, j) in pairs(f):
				neigh_collars[sortkey(i, j)] = []
				edge_perps[sortkey(i, j)] = np.zeros(3)
			
		for fi in range(len(faces)):
			f = faces[fi]
			edges = pairs(f)
			# find normal vector for the face and save it as edge normals
			# TODO: default edge_perps[(a,b)] to be average of all normal
			# vectors for faces with edge (a, b)
			n = face_normal(r0[f, :], ref=ref) # face normal
			x = np.mean(r0[f, :], axis=0) + z * n # collar node position

			for (i, j) in edges:
				edge_perps[sortkey(i, j)] += n / 2
			face_pos[fi] = x

		for fi in range(len(faces)):
			f = faces[fi]
			edges = pairs(f)
			edge_perps = {k: v / np.linalg.norm(v) \
				for (k, v) in edge_perps.items()}
			new_node = G.number_of_nodes()
			G.add_node(new_node, x0=face_pos[fi], cell=False)

			for i in f:
				G.add_edge(i, new_node, collar=True)

			for (i, j) in edges:
				neigh_collars[sortkey(i, j)].append(new_node)

		pos = nx.get_node_attributes(G, 'x0')

		for (k, v) in neigh_collars.items():
			assert len(v) > 0 and len(v) <= 2
			i = k[0]
			j = k[1]
			# handle cell-cell interactions with only one collar node
			if len(v) == 1:
				new_node = G.number_of_nodes()
				x = pos[v[0]] # existing collar position

				n = edge_perps[k] # normal of the face edge k is part of 
				closest = closest_point(pos[i], pos[j], pos[i] + n, x)
				x_to_plane = closest - x

				G.add_node(new_node, x0=(x + 2 * x_to_plane), cell=False)
				G.add_edge(i, new_node, collar=True)
				G.add_edge(j, new_node, collar=True)
				v.append(new_node)

			G.add_edge(i, j, collar=False, collar_pts=v)

		return(cls(G, **kwargs))

	@classmethod
	def trisurfgen(cls, r0,  **kwargs):
		"""Generates FlexaSheet object from arbitrary 3-column vector of 
		initial cell coordinates r0

		Args:
			x0 (np.ndarray): initial cell coordinates
			**kwargs: passed to __init__
		"""
		G = FlexaSheet.cellgraph(r0)
		tri = Delaunay(r0)

		# TODO: implement
		raise NotImplementedError('not done yet')

	@staticmethod
	def cellgraph(r0):
		"""Creates a graph with cell vertices with coordinates from r0"""
		G = nx.Graph()
		# add cell vertices
		G.add_nodes_from([(i, {'x0': r0[i, :], 'cell': True}) 
			for i in range(r0.shape[0])])
		return(G)

	@staticmethod
	def collar_pairs(neigh_collars):
		"""Returns an array of collar node pairs for cell-cell bdary ends"""
		collar_pairs = zip(list(neigh_collars.values()))
		collar_pairs = np.array(list(collar_pairs)).flatten().reshape(-1, 2)
		return(collar_pairs)
	
	def energy_stats(self, r):
		"""Prints energy statistics for sheet with coordinates r"""
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
		"""Prints average geometric statistics """
		print(
			  'Average phi: %0.3e\n' % self.aphi(r) + \
			  'Average psi: %0.3e\n' % self.apsi(r) + \
			  'Average collar length: %0.3e\n' % np.mean(self.collar_lengths(r))
			  )
	
	## network
	def cell_degrees(self):
		"""Returns dictionary {cell: number of neighboring cells}"""
		return({c: len(collars) for (c, collars) in self.cell_collars.items()})
	
	## simulation
	def cell_normal(self, r, cell):
		"""Returns normal vector corresponding to a cell based on its collars"""
		rcis = r[self.cell_collars[cell], :]
		rc = r[cell, :]
		n = np.sum(rcis - rc, axis=0)
		return n / np.linalg.norm(n)
		# return(face_normal(rcis, ref=np.sum(rcis - rc, axis=0)))
	
	def cell_normals(self, r):
		return {c: self.cell_normal(r, c) for c in self.cell_collars.keys()}

	# calculating the energies
	def phis(self, r, normals=None):
		"""Returns {cell: [phi angles for cell's collar nodes]}"""
		if normals is None:
			normals = self.cell_normals(r)
		ps = {}
		for (c, collar_nodes) in self.cell_collars.items():
			ps[c] = np.array([angle(normals[c], r[ci,:] - r[c,:]) \
			  for ci in collar_nodes])
		return(ps)
	
	# average phi
	def aphi(self, r):
		"""Returns average phi for whole sheet"""
		ps = self.phis(r)
		phi_tot = np.sum([p for cell_phis in ps.values() for p in cell_phis])
		n = sum(self.cell_degrees().values())
		return(phi_tot / n)
	
	def phi_energies(self, r, normals=None):
		"""Returns dictionary of phi energy for each cells"""
		return({i: np.sum((phis_c - self.phi0) ** 2) \
				for (i, phis_c) in self.phis(r, normals).items()})
	
	def phi_energy(self, r, normals=None):
		"""Returns phi energy for whole sheet"""
		e = sum(self.phi_energies(r, normals).values())
		return(e)
	
	def psis(self, r, normals=None):
		"""Returns dictionary {(cell1, cell2): psi}"""
		ps = {}
		if normals is None:
			normals = self.cell_normals(r)
		for (cells, collars) in self.neigh_collars.items():
			# cell normal 1
			n1 = normals[cells[0]]
			
			# find normal vector a for cell 1 to shared collar boundary
			c11 = r[collars[0], :] - r[cells[0], :]
			c12 = r[collars[1], :] - r[cells[0], :]
			a = np.cross(c11, c12) # collar-(cell 1)-collar normal
			a = align(a, n1) # align a with reference cell normal
	
			# repeat for cell 2
			c21 = r[collars[0], :] - r[cells[1], :]
			c22 = r[collars[1], :] - r[cells[1], :]
			n2 = normals[cells[1]]
			b = np.cross(c21, c22) # collar-(cell 2)-collar normal
			b = align(b, n2)
	
			# get the angle on the inside of the hinge between the two cells
			# at the collar boundary 
			psi = np.pi - angle(a, b)
			ps[cells] = psi / 2
		return(ps)
	
	def apsi(self, r):
		"""Returns average psi for whole sheet"""
		ps = self.psis(r)
		psi_tot = sum([p for p in ps.values()])
		n = len(self.neigh_collars)
		return(psi_tot / n)
	
	def psi_energies(self, r, normals=None):
		"""Returns {(cell1, cell2): psi energy = (psi - psi0) ** 2}"""
		return({uv: (psi - self.psi0) ** 2 \
				for (uv, psi) in self.psis(r, normals).items()})
	
	def psi_energy(self, r, normals=None):
		"""Returns whole sheet psi energy"""
		e = sum([(psi - self.psi0) ** 2 \
			for psi in self.psis(r, normals).values()])
		return(e)

	def collar_lengths(self, r):
		"""Returns 1d array of cell-collar lengths in order of collar_edges"""
		return(np.linalg.norm(r[self.collar_edges[:, 0], :] - \
							  r[self.collar_edges[:, 1], :], axis = 1))

	def spring_energy(self, r):
		"""Returns whole sheet cell-collar length energy"""
		return(np.sum((self.collar_lengths(r) - self.ell0) ** 2))

	def sector_angles(self, r, normals=None):
		if normals is None:
			normals = self.cell_normals(r)
		def angleacb(a, cell, b):
			return(angle(a - cell, b - cell))
		def sectors(e, v):
			c11 = closest_point_plane_eq(r[e[0], :], normals[e[0]], r[v[0]])
			c12 = closest_point_plane_eq(r[e[0], :], normals[e[0]], r[v[1]])

			c21 = closest_point_plane_eq(r[e[1], :], normals[e[1]], r[v[0]])
			c22 = closest_point_plane_eq(r[e[1], :], normals[e[1]], r[v[1]])
			return angleacb(c11, r[e[0], :], c12), \
				angleacb(c21, r[e[1], :], c22)
		vals = [sectors(e, v) for (e, v) in self.neigh_collars.items()]
		return np.array(vals)

	def sector_energy(self, r, normals=None):
		return(np.sum((self.sector_angles(r, normals) - self.s0) ** 2))

	def energy(self, x, k=0):
		"""Returns sum of whole sheet energies with spring energy times k"""
		r = x.reshape((-1, 3))
		normals = self.cell_normals(r)
		e = self.phi_energy(r, normals) + self.psi_energy(r, normals)
		e += k * self.spring_energy(r) 
		# e += self.sector_energy(r, normals)
		return(e)   
	 
	# simulate
	def solve_shape(self, k=0, silent=0):
		"""Minimise (with self.constrained) sheet energy (spring constant k)"""
		# silent 0: complete info, 1: time elapsed only, 2: nothing
		n_edges = self.collar_edges.shape[0]
			
		# defining the constraint in scipy format
		fixed_cell = 0 # fix a cell position
		r_fixed = self.x.reshape(-1, 3)[fixed_cell, :]
		n_fixed = self.cell_normal(self.x.reshape(-1, 3), fixed_cell)
		J_fixed_cell = np.zeros((3, 3 * self.n))
		J_fixed_cell[:, fixed_cell * 3 + np.arange(3)] = np.eye(3)

		eq_cons = []
		method = 'SLSQP'
		eq_cons = [{'type': 'eq',
			'fun': lambda x: x[fixed_cell * 3 + np.arange(3)] - r_fixed,
			'jac': lambda x: J_fixed_cell}]
		eq_cons.append({'type': 'eq',
			'fun': lambda x: self.cell_normal(x.reshape(-1, 3), fixed_cell) - n_fixed})
		# method = 'BFGS'
		if self.constrained:
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
			
			eq_cons.append({'type': 'eq',
				'fun': lambda x: 
					self.collar_lengths(x.reshape((-1, 3))) - self.ell0,
				'jac': lambda x: J_collar_lengths(x)})
			method = 'SLSQP'
			assert k == 0
		# blasting it with the minimisation routine
		if silent == 0:
			print('\nBeginning solver')
		t = time.time()
		res = minimize(
				lambda x: self.energy(x, k), self.x, 
				method=method, 
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
		"""Plots phi and psi energies as functions of radius out from x,y=0"""
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
	def draw(self, style='flat', x0=False, nodes=[], edges=[], 
			linecolor=np.array([0, 0, 1]), collarcolor=np.array([1, 0, 0]),
			ax=None):
		"""Plotting method for flexa sheets. 

		Args:
			style (str, optional): choice of style. Currently supports 'flat' 
				and '3d'. Defaults to 'flat'.
			x0 (bool, optional): option to plot based on initial coordinates 
				rather than current coordinates `x`. Defaults to False.
			nodes (list, optional): list of nodes to highlight. Defaults to [].
			edges (list, optional): list of edges to highlight. Defaults to [].
			linecolor (np.ndarray): array of length 3 for collar-collar
				line color. Only used when style='3d'. Defaults to blue,
				np.array([0, 0, 1])
			collarcolor (np.ndarray): array of length 3 for cell-collar
				line color. Only used when style='3d'. Defaults to red,
				np.array([1, 0, 0])
			ax (_type_, optional): axes to plot on. If specified and 
				`style='3d'`, only one projection is shown. Defaults to None.
		"""
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
			def plot_center(ax):
				"""Returns the center of the axes ax"""
				center = np.array([np.mean(ax.get_xlim3d()),
								   np.mean(ax.get_ylim3d()),
								   np.mean(ax.get_zlim3d())])
				return(center)
			
			def linecolors(r, origin, azim, elev, color, maxalpha=1, rev=False):
				"""Generate color argument for LineCollection based on closeness
				of vectors r to the camera. The cos of the angle between the 
				camera and vectors r is used as the alpha value. 

				Args:
					r (np.ndarray): (n, 3) array of vectors
					origin (np.ndarray): length 3 array for the origin coords
					azim (float): azimuthal camera angle
					elev (float): elevation camera angle
					color (np.ndarray): length 3 array for rgb color
					maxalpha (float): between 0 and 1, maximum alpha value
				"""
				n = r.shape[0]
				ors = r - origin # vectors o to r
				oc = np.array([np.cos(elev) * np.cos(azim), 
							np.cos(elev) * np.sin(azim),
							np.sin(elev)])[:, np.newaxis] # vector o to camera

				alphas = (ors @ oc) / \
					np.linalg.norm(ors, axis=1)[:, np.newaxis] / \
					np.linalg.norm(oc) # = cos(angle between or and oc)
				alphas = maxalpha * (alphas - alphas.min()) / np.ptp(alphas)
				colors = np.concatenate((np.tile(color, (n, 1)), alphas),
					axis=1)
				if rev:
					colors[:, 3] = (1 - colors[:, 3]) - (1 - maxalpha)
				return(colors)

			def colored_line_collection(r, pairs, color, ax, maxalpha=1, 
					rev=False):
				"""Returns a Line3DCollection between pairs from vectors r
				with color and alpha determined by distance from the camera.

				Args:
					r (np.ndarray): position vector
					pairs (np.ndarray): (n, 2) array of indices
					color (np.ndarray): 3 vector passed to linecolors
					ax (matplotlib axes): axes to plot on
					maxalpha (float, optional): between 0 and 1. Defaults to 1.
					rev (bool, optional): option to reverse alphas,
						so closer to camera is more transparent. Defaults to 
						False.
				"""
				xflines = r.reshape(-1, 1, 3) # format for lines, idk why
				
				azim = np.deg2rad(ax.azim)
				elev = np.deg2rad(ax.elev)
				
				colors = linecolors((r[pairs[:, 0]] + r[pairs[:, 1]]) / 2, 
					plot_center(ax), azim, elev, color, maxalpha, rev)

				c = Line3DCollection(
					np.concatenate(
						[xflines[pairs[:, 1]], 
						 xflines[pairs[:, 0]]], axis = 1),
					color=colors)
				return(c)

			fig = plt.gcf()
			r = self.x.reshape((-1, 3))
			
			# first subplot
			ax_given = True
			if ax is None:
				ax = fig.add_subplot(1, 2, 1, projection='3d')
				ax_given = False
			ax.view_init(10, 20)
			
			# extra plotting options that aren't currently used
			'''# plot cell-cell connections
			ax.plot_trisurf(r[:self.n_cells, 0], 
							r[:self.n_cells, 1], 
							r[:self.n_cells, 2],
							cmap='spring', edgecolor='black', alpha=0)
			
			# plot cell bodies
			cs = linecolors(r[:self.n_cells, :], plot_center(ax), 
				np.deg2rad(ax.azim), np.deg2rad(ax.elev), 
				color=np.array([1, 0, 0]), maxalpha=0.5, rev=True)
			ax.scatter3D(r[:self.n_cells, 0], 
						 r[:self.n_cells, 1], 
						 r[:self.n_cells, 2],
						 c=cs, s=300)
			'''

			# plot cell-collar connections
			c = colored_line_collection(r, self.collar_edges, collarcolor, ax,
				maxalpha=0.1, rev=True)
			ax.add_collection3d(c)

			# plot collar boundaries
			ax.scatter3D(r[self.n_cells:, 0], # pyplot handles the alpha
						 r[self.n_cells:, 1], # on this one by itself
						 r[self.n_cells:, 2])

			# plot collar boundaries
			c = colored_line_collection(r, 
				FlexaSheet.collar_pairs(self.neigh_collars), 
				linecolor, ax,)
			ax.add_collection3d(c)
			if ax_given:
				return(ax)
			
			# second subplot
			ax = fig.add_subplot(1, 2, 2, projection='3d')
			ax.view_init(40, 20)

			# matplotlib won't let me reuse these artists! >:(
			c = colored_line_collection(r, self.collar_edges, collarcolor, ax,
				maxalpha=0.1, rev=True)
			ax.add_collection3d(c)
			c = colored_line_collection(r, 
				FlexaSheet.collar_pairs(self.neigh_collars), 
				linecolor, ax,)
			ax.add_collection3d(c)
			ax.scatter3D(r[self.n_cells:, 0],
						 r[self.n_cells:, 1],
						 r[self.n_cells:, 2])

			plt.tight_layout()
		# TODO: add plotter showing phi energy at each cell and 
		# psi energy at each boundary
	
	## file management
	def __eq__(self, other):
		"""Tests FlexaSheet equality based on coordinates and edges"""
		if isinstance(other, FlexaSheet):
			return np.all(self.x == other.x) and \
				self.cell_collars == other.cell_collars and \
				self.neigh_collars == other.neigh_collars
		return(False)

	def save(self, name):
		"""Efficiently stores this FlexaSheet object at filepath name"""
		data = {}

		data['init_params'] = {'G': self.G, 'phi0': self.phi0, 
			'psi0': self.psi0, 'ell0': self.ell0, 
			'constrained': self.constrained}
		
		data['attributes'] = {'x': self.x}

		with open(picklify(name), 'wb') as f:
			pickle.dump(data, f)

	@classmethod
	def load(cls, name, silent=1):
		"""Loads FlexaSheet at filepath name, passes silent to __init__"""
		with open(picklify(name), 'rb') as f:
			saved = pickle.load(f)
		
		data = saved['init_params']
		s = cls(data['G'], data['phi0'], data['psi0'], data['ell0'],
			data['constrained'], silent)
		
		s.__dict__.update(saved['attributes'])
		
		return(s)
