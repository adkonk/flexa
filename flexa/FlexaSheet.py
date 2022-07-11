import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial import Voronoi, Delaunay, SphericalVoronoi
from scipy.optimize import minimize
from scipy.stats import rankdata
import time
import pickle
import os.path

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
		s0 (np.ndarray): 2-column array of equilibrium collar-cell-collar angles
			based on self.neigh_collars [(angle collar1-cell1-collar2,
			angle collar1-cell2-collar2)]
		normals (str): option for calculating cell normals
	"""
	def __init__(self, G, phi0=None, psi0=None, ell0=None, normals='free',
			silent=0):
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
			normals (str, optional): option for calculating cell normals.
				Defaults to 'free'.
				Options:
					'avg': uses the unit vector in the direction of average
						cell-to-collar vector
					'lsc': means Least Squares Collar. Finds a least squares
						plane approximation to collars belonging to each cell
						and uses the normal vector as the cell normal
					'free': cell normals are their own variable to be
						optimised
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

		# TODO: be certain this is always (cell, collar node)
		# list of (cell, collar node) or (collar node, cell)
		self.collar_edges = np.array(
			[(e[0], e[1]) for e in self.G.edges.data('collar') if e[2]])

		self.normals = normals
		if self.normals == 'free':
			n = np.array(list(self.cell_normals(r, mode='lsc').values()))
			r = np.append(r, n, axis=0)
			self.x = np.append(self.x, n.flatten())

		if phi0 is None: # phi0 = average initial phi0
			self.phi0 = self.aphi(r)
		else: self.phi0 = phi0
		if psi0 is None:# psi0 = average initial psi0
			self.psi0 = self.apsi(r)
		else: self.psi0 = psi0
		if ell0 is None: # ell0 = initial ell0s
			self.ell0 = self.collar_lengths(r)
		else: self.ell0 = ell0
		self.s0 = self.sector_angles(r)

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
	def cell_degree(self, c):
		if isinstance(c, int):
			return len(self.cell_collars[c])
		else:
			return [len(self.cell_collars[cell]) for cell in c]

	def cell_normal(self, r, cell, mode=None):
		"""Returns normal vector corresponding to a cell based on its collars"""
		if mode is None:
			mode = self.normals

		if mode == 'free':
			assert self.normals == 'free', "free normals must be set from init"
			return r[self.n + cell, :]

		rcis = r[self.cell_collars[cell], :]
		rc = r[cell, :]
		if mode == 'avg':
			n = np.sum(rcis - rc, axis=0)
			return n / np.linalg.norm(n)
		elif mode == 'lsc':
			return(face_normal(rcis, ref=np.sum(rcis - rc, axis=0)))
		else:
			raise NotImplementedError('mode are written yet')

	def cell_normals(self, r, mode=None):
		return {c: self.cell_normal(r, c, mode) \
			for c in self.cell_collars.keys()}

	def _cell_normals_array(self, r, mode=None, order=None, normals=None):
		if order is None:
			order = np.arange(self.n_cells)
		if normals is not None:
			return np.array([normals[c] for c in order])

		if mode is None and self.normals == 'free':
			return self.r[self.n:, :][order, :]
		return np.array([self.cell_normal(r, c, mode) for c in order])

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
		return({c: np.sum((phis_ci - self.phi0) ** 2) \
				for (c, phis_ci) in self.phis(r, normals).items()})

	def _phis_array(self, r, normals=None):
		"""Returns phis corresponding to rows in self.collar_edges[:, 0]"""
		ps = np.zeros(self.collar_edges.shape[0])
		if normals is None:
			normals = self.cell_normals(r)
		for i in range(len(ps)):
			c, ci = self.collar_edges[i, :]
			ps[i] = angle(normals[c], r[ci,:] - r[c,:])
		return ps

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

	def _psis_array(self, r, normals=None):
		ps = self.psis(r, normals)
		return np.array([ps[cells] for cells in self.neigh_collars.keys()])

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
		"""Returns collar-collar angles when projected on cell normal plane"""
		if normals is None:
			normals = self.cell_normals(r)

		def angleacb(a, cell, b): # angle between three points at cell
			return(angle(a - cell, b - cell))
		def sectors(e, v): #e: (cell, cell) inds, v: (collar, collar) inds
			c11 = closest_point_plane_eq(r[e[0], :], normals[e[0]], r[v[0]])
			c12 = closest_point_plane_eq(r[e[0], :], normals[e[0]], r[v[1]])

			c21 = closest_point_plane_eq(r[e[1], :], normals[e[1]], r[v[0]])
			c22 = closest_point_plane_eq(r[e[1], :], normals[e[1]], r[v[1]])
			return angleacb(c11, r[e[0], :], c12), \
				angleacb(c21, r[e[1], :], c22)
		vals = [sectors(e, v) for (e, v) in self.neigh_collars.items()]
		return np.array(vals) # array of (angle pas, pbs)

	def sector_energy(self, r, normals=None):
		return(np.sum((self.sector_angles(r, normals) - self.s0) ** 2))

	def energy(self, x, k=(1, 2, 0)):
		"""Returns sum of whole sheet energies times energy constants k"""
		r = x.reshape((-1, 3))
		normals = self.cell_normals(r)
		e = k[0] * self.phi_energy(r, normals) + \
			k[1] * self.psi_energy(r, normals) + \
			k[2] * self.spring_energy(r) + \
			0.1 * k[0] * self.sector_energy(r, normals)
		return(e)

	# simulate
	def solve_shape(self, k=(1, 2, 0), constrained=False, silent=0):
		"""Minimise (with constrained) sheet energy (energy constants k)"""
		# silent 0: complete info, 1: time elapsed only, 2: nothing
		n_edges = self.collar_edges.shape[0]

		# TODO: update numerical optimisation for free cell normals

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
		if constrained:
			if constrained: # all initial lengths must == ell0 if constr
				assert np.all(
					self.collar_lengths(self.x.reshape(-1, 3)) == self.ell0)
			eq_cons = {'type': 'eq',
					   'fun': lambda x:
						   self.collar_lengths(x.reshape((-1, 3))) - self.ell0,
					   'jac': lambda x: J_collar_lengths(x)}
			assert k[2] == 0
		else: eq_cons = []
		# blasting it with the minimisation routine
		if silent == 0:
			print('\nBeginning solver')
		t = time.time()
		res = minimize(
				self.energy, self.x, method='SLSQP',
				args = (k, ),
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

	def _graph_delta_mxs(self):
		# (self.n, n_cc) kronecker deltas
		# can also do (arange[:, None] == cells[None, :])
		dga_cc = np.equal.outer(
			np.arange(self.n), self.collar_edges[:, 0]).astype('int')
		dgp_cc = np.equal.outer(
			np.arange(self.n), self.collar_edges[:, 1]).astype('int')

		degs = self.cell_degrees()
		degs = np.array([degs[c] for c in self.collar_edges[:, 0]])

		# need to make array with rows of self.cell_collars but
		# self.cell_collars is ragged so fill all the extra entries with
		# self.n, which we know will not be equal to any element in
		# np.arange(self.n)
		collar_inds = np.full((self.collar_edges.shape[0], np.amax(degs)),
			self.n)
		for ci in np.arange(self.collar_edges.shape[0]):
			c = self.collar_edges[ci, 0]
			collar_inds[ci, :len(self.cell_collars[c])] = self.cell_collars[c]

		is_collar_in_cell = \
				(np.arange(self.n)[:, np.newaxis, np.newaxis] == \
				collar_inds[np.newaxis, :, :]).sum(-1)

		# array of (cell1, cell2, collar1, collar2)
		abps = np.array([(cells[0], cells[1], collars[0], collars[1]) for \
			(cells, collars) in self.neigh_collars.items()])
		dga = np.equal.outer(np.arange(self.n), abps[:, 0]).astype('int')
		dgb = np.equal.outer(np.arange(self.n), abps[:, 1]).astype('int')
		dgp = np.equal.outer(np.arange(self.n), abps[:, 2]).astype('int')
		dgs = np.equal.outer(np.arange(self.n), abps[:, 3]).astype('int')

		deltas = [degs, dga_cc, dgp_cc, is_collar_in_cell, abps, dga, dgb, \
			dgp, dgs]

		if self.normals == 'free':
			collars_to_cells = np.arange(self.n_cells)[:, np.newaxis] == \
				self.collar_edges[:, 0][np.newaxis, :]
			deltas.append(collars_to_cells.astype('int'))

		return deltas

	def grad(self, k, deltas=None):
		""" Computes gradient of energy with energy constants k"""
		# Naming convention:
		# 	a, b: cell indices
		# 	p, s: collar indices
		# 	n: cell normal vector
		# 	m: collar-cell-collar normal vector
		#	_cc: cell-collar (columns indexed by cell-collar pair as in rows
		# 		self.collar_edges)

		if deltas is None:
			deltas = self._graph_delta_mxs()

		r = self.x.reshape(-1, 3)
		degs = deltas[0]
		if self.normals == 'lsc':
			# (number of cells, max number of collar nodes to a cell, 3)
			# first axis needs to be number of cell-collar bonds, but we will
			# just use indexing later to get this so we don't calculate
			# redundant inverses
			X = np.zeros((self.n_cells, np.amax(degs), 3))
			# (n_cells, d_max)
			z = np.zeros((X.shape[0], X.shape[1]))
			for i in range(X.shape[0]):
				n = degs[i]
				X[i, :n, :2] = r[self.cell_collars[i], :2]
				X[i, :n, 2] = np.ones(n)
				z[i, :n] = r[self.cell_collars[i], 2]
			# (n_cells, 3, 3)
			XtXinv = np.linalg.inv(X.swapaxes(1, 2) @ X)
			# (n_cells, 3, 3) @ (n_cells, 3, d_max) @ (n_cells, d_max, 1)
			# --> (n_cells, 3, 1) --- squeeze ---> (n_cells, 3)
			XtXinvXtz = np.squeeze(XtXinv @ X.swapaxes(1, 2) @ \
				z[:, :, np.newaxis])
			nvec = XtXinvXtz
			nvec[:, 2] = -1 # can get rid of the constant offset estimation

			ref = self._cell_normals_array(r, mode='avg') # (n_cells, 3)

			nvec *= np.sign(rowwise_dot(nvec, ref))[:, np.newaxis]

			nvec_norm = np.linalg.norm(nvec, axis=1)
			normals = {i: nvec[i, :] / nvec_norm[i, :] for i in \
				range(nvec.shape[0])}
		else: normals = self.cell_normals(r)

		## dphiE / dr
		# d(ap_hat)_j / dr dot n, this is always the same for all normals

		# cell to collar vectors \vec{ap} array (n_cc, 3)
		ap = r[self.collar_edges[:, 1], :] - r[self.collar_edges[:, 0], :]
		ap_norm = np.linalg.norm(ap, axis=1, keepdims=True)
		ap_hat = ap / ap_norm

		n = self._cell_normals_array(r, normals=normals,
			order=self.collar_edges[:, 0])

		# (n_cc, ) factor in dap/dr dot n
		dphi = -1 * (self._phis_array(r, normals=normals) - self.phi0) / \
			np.sqrt(1 - rowwise_dot(n, ap_hat) ** 2)

		# d\vec{ap}/d\vec{r}_gamma dot \vec{n}_gamma for all gamma
		dapdr_n = 1 / ap_norm * \
			(n - ap_hat * rowwise_dot(n, ap_hat, keepdims=True))
		dapdr_n *= dphi[:, np.newaxis]

		dga_cc = deltas[1]
		dgp_cc = deltas[2]

		# (self.n, n_cc) * (n_cc, 3) --> (self.n, 3) gradient
		dphiE_dr = np.matmul(dgp_cc - dga_cc, dapdr_n)

		# d(n)_j / dr dot ap_hat

		# (self.n, n_cc) array with 1s whenever rowindex is in
		# the list self.cell_collars[rowindex]
		#
		# (self.n, 1, 1) * (1, n_cc, d_max) --> (self.n, n_cc, d_max)
		# --- sum over last axis ---> (self.n, n_cc)
		is_collar_in_cell = deltas[3]
		if self.normals == 'lsc':
			# remake collar_inds to make ind_ga because I don't want to include
			# it in deltas
			collar_inds = np.full((self.collar_edges.shape[0], np.amax(degs)),
				self.n)
			for ci in np.arange(self.collar_edges.shape[0]):
				c = self.collar_edges[ci, 0]
				collar_inds[ci, :len(self.cell_collars[c])] = \
					self.cell_collars[c]
			# (n, n_cc, d_max)
			ind_ga = np.arange(self.n)[:, np.newaxis, np.newaxis] == \
				collar_inds[np.newaxis, :, :]

		if self.normals == 'avg':
			norms = {c: np.linalg.norm(np.sum(r[ci, :] - r[c, :], axis=0)) \
				for (c, ci) in self.cell_collars.items()}
			denom = np.array([norms[c] \
				for c in self.collar_edges[:, 0]])[:, np.newaxis]
			dndr_aphat = 1 / denom * \
				(ap_hat - n * (rowwise_dot(n, ap_hat, keepdims=True)))

			dndr_aphat *= dphi[:, np.newaxis]
			dphiE_dr += np.matmul(-dga_cc * degs + is_collar_in_cell,
				dndr_aphat)
		elif self.normals == 'lsc':
			# (n, n_cc, 2, 3)
			# axes:
			#   0: which node (gamma, preserved until the end)
			#   1: which cell-collar bond (summed with dphi)
			#   2: which beta (summed with betas)
			#   3: which component of r_gamma (x, y, z)
			db_dr = np.zeros((self.n, ap.shape[0], 2, 3))

			## derivative wrt r_gamma_x
			# d XtXinv / dr_gamma_x
			dXtXinv_drg0 = np.zeros((self.n, 3, 3))
			dXtXinv_drg0[:, 0, 0] = 2 * r[:, 0]
			dXtXinv_drg0[:, [1, 0], [0, 1]] = r[:, [1]] # keep column dim
			dXtXinv_drg0[:, [2, 0], [0, 2]] = 1

			# (n, 3)
			dXt_drg0 = np.zeros((self.n, 3))
			dXt_drg0[:, 0] = r[:, 2]

			# (n_cc, 2, 3)_ijk *
			# [(n, n_cc) * (n, 3) --> (n, n_cc, 3) -
			#   {(n, n_cc) * (n, 3, 3) --> (n, n_cc, 3, 3) *
			#   (n_cc, 3)} --> (n, n_cc, 3)]_lik
			# gets you (n, n_cc, 2)
			db_dr[..., 0] = np.einsum('ijk,lik->lij',
				XtXinv[self.collar_edges[:, 0], :2, :],
				is_collar_in_cell[..., np.newaxis] * \
					dXt_drg0[:, np.newaxis, :] - \
					np.squeeze(
						(is_collar_in_cell[..., np.newaxis, np.newaxis] * \
						 dXtXinv_drg0[:, np.newaxis, ...]) @ \
						XtXinvXtz[self.collar_edges[:, 0], :, np.newaxis])
			)

			## derivative wrt r_gamma_y
			# d XtXinv / dr_gamma_y
			dXtXinv_drg1 = np.zeros(dXtXinv_drg0.shape)
			dXtXinv_drg1[:, 1, 1] = 2 * r[:, 1]
			dXtXinv_drg1[:, [1, 0], [0, 1]] = r[:, [0]] # keep column dim
			dXtXinv_drg1[:, [1, 2], [2, 1]] = 1

			# (n, 3)
			dXt_drg1 = np.zeros((self.n, 3))
			dXt_drg1[:, 1] = r[:, 2]

			# (n_cc, 2, 3)_ijk *
			# [(n, n_cc) * (n, 3) --> (n, n_cc, 3) -
			#   {(n, n_cc) * (n, 3, 3) --> (n, n_cc, 3, 3) *
			#   (n_cc, 3)} --> (n, n_cc, 3)]_lik
			# gets you (n, n_cc, 2)
			db_dr[..., 1] = np.einsum('ijk,lik->lij',
				XtXinv[self.collar_edges[:, 0], :2, :],
				is_collar_in_cell[..., np.newaxis] * \
					dXt_drg1[:, np.newaxis, :] - \
					np.squeeze(
						(is_collar_in_cell[..., np.newaxis, np.newaxis] * \
						 dXtXinv_drg1[:, np.newaxis, ...]) @ \
						XtXinvXtz[self.collar_edges[:, 0], :, np.newaxis])
			)

			## derivative wrt r_gamma_z
			db_dr[..., 2] = np.einsum('ijk,lik->lij',
				XtXinv[self.collar_edges[:, 0], :2, :] @ \
					X.swapaxes(1, 2)[self.collar_edges[:, 0], ...],
				ind_ga)

			# (n, n_cc, 2, 3) --> (n, n_cc, 3vec, 3) padded with zeros
			# 3vec axis to be summed with ap
			dn_dr_part1 = np.pad(db_dr / \
					nvec_norm[np.newaxis, self.collar_edges[:, 0], \
						np.newaxis, np.newaxis],
				((0, 0), (0, 0), (0, 1), (0, 0)))

			dn_dr_part2 = np.einsum('ij,kil->kijl',
				# (n_cc, 3vec)
				nvec[self.collar_edges[:, 0], :] / \
					nvec_norm[self.collar_edges[:, 0], np.newaxis] ** 3,
				# (n, n_cc, 3)
				np.einsum('ik,likm->lim',
					XtXinvXtz[self.collar_edges[:, 0], :2],
					db_dr)
			)

			# (n, 3)
			dphiE_dr += np.einsum('ij,kijl->kl',
				dphi[:, np.newaxis] * ap_hat, # (n_cc, 3vec)
				dn_dr_part1 - dn_dr_part2) # (n, n_cc, 3vec, 3)
		elif self.normals == 'free':
			# the only term in dphiE_dr for coordinates r_gamma (up to row
			# self.n) is d(ap)_dr dot n_a since n_a is a free variable
			#
			# we will add the (n_cells, 3) cell normal rows at the end
			pass

		## dspringE_dr
		# Hooke's law: (r - r0) * rhat
		dell = (ap_norm - self.ell0) * ap_hat
		dspringE_dr = np.matmul(dgp_cc - dga_cc, dell)

		## dpsiE_dr
		abps, dga, dgb, dgp, dgs = deltas[4:9]

		na = np.array([normals[c] for c in abps[:, 0]])
		nb = np.array([normals[c] for c in abps[:, 1]])

		rap = r[abps[:, 2], :] - r[abps[:, 0], :]
		ras = r[abps[:, 3], :] - r[abps[:, 0], :]
		rbp = r[abps[:, 2], :] - r[abps[:, 1], :]
		rbs = r[abps[:, 3], :] - r[abps[:, 1], :]

		psis = self._psis_array(r, normals=normals)[:, np.newaxis]

		npas_long = np.cross(rap, ras)
		norm_npas = np.linalg.norm(npas_long, axis=1)
		Ypas = (np.sign(rowwise_dot(npas_long, na)) / norm_npas)[:, np.newaxis]
		npas = npas_long * Ypas

		npbs_long = np.cross(rbp, rbs)
		norm_npbs = np.linalg.norm(npbs_long, axis=1)
		Ypbs = (np.sign(rowwise_dot(npbs_long, nb)) / norm_npbs)[:, np.newaxis]
		npbs = npbs_long * Ypbs

		Ypas *= (psis - self.psi0) / \
			np.sqrt(1 - rowwise_dot(npas, npbs, keepdims=True)**2)
		Ypbs *= (psis - self.psi0) / \
			np.sqrt(1 - rowwise_dot(npas, npbs, keepdims=True) ** 2)

		Xb = rowwise_dot(npas, npbs_long, keepdims=True) / \
			norm_npbs[:, np.newaxis] ** 2
		bvec = Ypbs * (Xb * npbs_long - npas)
		dpsiE_dr = np.matmul(dgp - dgb, np.cross(bvec, rbs))
		dpsiE_dr -= np.matmul(dgs - dgb, np.cross(bvec, rbp))

		Xa = rowwise_dot(npbs, npas_long, keepdims=True) / \
			norm_npas[:, np.newaxis] ** 2
		avec = Ypas * (Xa * npas_long - npbs)
		dpsiE_dr += np.matmul(dgp - dga, np.cross(avec, ras))
		dpsiE_dr -= np.matmul(dgs - dga, np.cross(avec, rap))

		dpsiE_dr *= 1 / 2 # from the psi = pi / 2 - 1 / 2 arccos
		# the -'s from the arccos derivative and -1/2 have already been included

		## dsecE_dr
		# TODO: this only works if self.normals == 'free'
		ap2 = rap - rowwise_dot(rap, na, keepdims=True) * na
		as2 = ras - rowwise_dot(ras, na, keepdims=True) * na
		bp2 = rbp - rowwise_dot(rbp, nb, keepdims=True) * nb
		bs2 = rbs - rowwise_dot(rbs, nb, keepdims=True) * nb

		ap2_norm = np.linalg.norm(ap2, axis=1, keepdims=True)
		as2_norm = np.linalg.norm(as2, axis=1, keepdims=True)
		bp2_norm = np.linalg.norm(bp2, axis=1, keepdims=True)
		bs2_norm = np.linalg.norm(bs2, axis=1, keepdims=True)

		ap2 /= ap2_norm
		as2 /= as2_norm
		bp2 /= bp2_norm
		bs2 /= bs2_norm

		S = self.sector_angles(r) # (n_neigh, 2)
		dSa = -(S[:, [0]] - self.s0[:, [0]]) / \
			np.sqrt(1 - rowwise_dot(ap2, as2, keepdims=True) ** 2)
		dSb = -(S[:, [1]] - self.s0[:, [1]]) / \
			np.sqrt(1 - rowwise_dot(bp2, bs2, keepdims=True) ** 2)

		dsecE_dr = np.matmul( # d(ap2)/dr dot as2
			dgp - dga,
			dSa / ap2_norm * (as2 - rowwise_dot(as2, na, keepdims=True) * na - \
				rowwise_dot(ap2, as2, keepdims=True) * ap2 + \
				rowwise_dot(ap2, na, keepdims=True) * \
					rowwise_dot(ap2, as2, keepdims=True) * na)
		)

		dsecE_dr += np.matmul( # d(as2)/dr dot ap2
			dgs - dga,
			dSa / as2_norm * (ap2 - rowwise_dot(ap2, na, keepdims=True) * na - \
				rowwise_dot(ap2, as2, keepdims=True) * as2 + \
				rowwise_dot(as2, na, keepdims=True) * \
					rowwise_dot(ap2, as2, keepdims=True) * na)
		)

		dsecE_dr += np.matmul( # d(bp2)/dr dot bs2
			dgp - dgb,
			dSb / bp2_norm * (bs2 - rowwise_dot(bs2, nb, keepdims=True) * nb - \
				rowwise_dot(bp2, bs2, keepdims=True) * bp2 + \
				rowwise_dot(bp2, nb, keepdims=True) * \
					rowwise_dot(bp2, bs2, keepdims=True) * nb)
		)

		dsecE_dr += np.matmul( # d(bs2)/dr dot bp2
			dgs - dgb,
			dSb / bs2_norm * (bp2 - rowwise_dot(bp2, nb, keepdims=True) * nb - \
				rowwise_dot(bp2, bs2, keepdims=True) * bs2 + \
				rowwise_dot(bs2, nb, keepdims=True) * \
					rowwise_dot(bp2, bs2, keepdims=True) * nb)
		)

		dsecE_dn = np.matmul( # -1 is included in dSa
			dga[:self.n_cells, :],
			-dSa * (rowwise_dot(rap, na, keepdims=True) * as2 + \
				rap * rowwise_dot(na, as2, keepdims=True) + \
				ap2 * rowwise_dot(ras, na, keepdims=True) + \
				ras * rowwise_dot(na, ap2, keepdims=True))
		)

		dsecE_dn += np.matmul( # -1 is i-1 * ncluded in dSb
			dgb[:self.n_cells, :],
			-dSb * (rowwise_dot(rbp, nb, keepdims=True) * bs2 + \
				rbp * rowwise_dot(nb, bs2, keepdims=True) + \
				bp2 * rowwise_dot(rbs, nb, keepdims=True) + \
				rbs * rowwise_dot(nb, bp2, keepdims=True))
		)

		dsecE_dr = np.concatenate((dsecE_dr, dsecE_dn), axis=0)

		if self.normals == 'free':
			collars_to_cells = deltas[9]

			dphiE_dr = np.append(dphiE_dr,
				np.matmul(collars_to_cells, dphi[:, np.newaxis] * ap_hat),
				axis=0)
			dpsiE_dr = np.append(dpsiE_dr,
				np.zeros((self.n_cells, 3)), axis=0)
			dspringE_dr = np.append(dspringE_dr,
				np.zeros((self.n_cells, 3)), axis=0)

		return k[0] * dphiE_dr + \
			k[1] * dpsiE_dr + \
			k[2] * dspringE_dr + \
			0.1 * k[0] * dsecE_dr

	def f_equil(self, k, rate=1e-2, tol=1e-4, silent=1,
			plot=False, plotint=10, plotdir=None, m=0):
		"""Performs (projected) gradient descent on the coordinates in self.x.
		If self.normals == 'free', then the updated (not necessarily unit) cell
		normal vectors are projected onto the constraint set where the cell
		normals are normalised.

		Prints a message if the energy goes up.

		Args:
			k (tuple): tuple of energy constants for phi, psi, spring
			rate (float, optional): rate of gradient descent. Defaults to 1e-2.
			tol (float, optional): relative change in energy to stop converging.
				Defaults to 1e-4.
			silent (int, optional): is algorithm silent or not. 0 to print out
				details at every plotint and 1 to be silent. Defaults to 1.
			plot (bool, optional): option to plot frames at interval plotint.
				Defaults to False.
			plotint (int, optional): plotting interval counted in number of
				steps. Defaults to 10.
			plotdir (str, optional): directory to save plots. Defaults to None.
			m (int, optional): initial frame number, used for plotting movies
				with several equilibrations in a row. Defaults to 0.

		Returns:
			number of steps before convergence
		"""
		if plot:
			fig = plt.gcf()
			fig.clear()
			fig.set_size_inches(6, 6)
			ax = fig.add_subplot(1, 1, 1, projection='3d')

		t = time.time()

		# matrices describing graph topology that don't change each step
		deltas = self._graph_delta_mxs()

		e_new = self.energy(self.x, k)
		e_old = e_new * (1 + 2 * tol)
		n_steps = m

		while (e_old - e_new) / e_old > tol:
			g = self.grad(k, deltas=deltas)
			self.x -= rate * g.flatten()
			if self.normals == 'free':
				# normalise free cell normals
				self.x[(3 * self.n):] /= \
					np.repeat(np.linalg.norm(
						self.x[(3 * self.n):].reshape(-1, 3), axis=1), 3)
			e_old = e_new
			e_new = self.energy(self.x, k)
			n_steps += 1
			if plot and ((n_steps % plotint == 0) or n_steps == 1):
				ax.clear()
				ax.set_xlim([-1.5, 1.5])
				ax.set_ylim([-1.5, 1.5])
				ax.set_zlim([-2, 1])
				self.draw('3d', ax=ax)
				ax.set_title(r'$t=%0.4f$' % (n_steps * rate), fontsize=16)
				fig.tight_layout()
				plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
				fig.savefig(
					os.path.join(plotdir,
						'%05d.png' % (int(n_steps / plotint))),
					dpi=200, bbox_inches='tight')
			if (silent == 0) and (n_steps % plotint == 0):
				diff = (e_old - e_new) / e_old
				print('Step %d, energy %0.5f, decrease %0.2e' % \
					(n_steps, e_new, diff))
			if (e_old - e_new) / e_old < 0:
				print('increase in energy!')

		if silent in [0, 1]:
			print('number of steps: ', n_steps)
			print('time elapsed: %0.2f minutes' % ((time.time() - t) / 60))
		return(n_steps)

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
				# format for lines, idk why
				xflines = r.reshape(-1, 1, 3)

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

			# plot cell-collar connections
			c = colored_line_collection(r[:self.n, :], self.collar_edges,
				collarcolor, ax, maxalpha=0.1, rev=True)
			ax.add_collection3d(c)

			# plot cell normals
			if self.normals == 'free':
				colors = c.get_edgecolors()
				colors[:, 3] **= 0.3
				ax.quiver(*r[:self.n_cells, :].T, *r[self.n:, :].T,
					length=0.2, color=colors)

			# plot collar boundaries
			ax.scatter3D(r[self.n_cells:self.n, 0], # pyplot handles the alpha
						 r[self.n_cells:self.n, 1], # on this one by itself
						 r[self.n_cells:self.n, 2])

			# plot collar boundaries
			c = colored_line_collection(r[:self.n, :],
				FlexaSheet.collar_pairs(self.neigh_collars),
				linecolor, ax,)
			ax.add_collection3d(c)

			if ax_given:
				return(ax)

			# second subplot
			ax = fig.add_subplot(1, 2, 2, projection='3d')
			ax.view_init(40, 20)

			# matplotlib won't let me reuse these artists! >:(
			c = colored_line_collection(r[:self.n, :], self.collar_edges,
				collarcolor, ax, maxalpha=0.1, rev=True)
			ax.add_collection3d(c)

			if self.normals == 'free':
				colors = c.get_edgecolors()
				colors[:, 3] **= 0.3
				ax.quiver(*r[:self.n_cells, :].T, *r[self.n:, :].T,
					length=0.2, color=colors)

			ax.scatter3D(r[self.n_cells:self.n, 0],
						 r[self.n_cells:self.n, 1],
						 r[self.n_cells:self.n, 2])

			c = colored_line_collection(r[:self.n, :],
				FlexaSheet.collar_pairs(self.neigh_collars),
				linecolor, ax,)
			ax.add_collection3d(c)

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
			'normals': self.normals}

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
			data['normals'], silent)

		s.__dict__.update(saved['attributes'])

		return(s)
