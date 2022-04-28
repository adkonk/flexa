from flexa.FlexaSheet import FlexaSheet
import unittest

import numpy as np
import networkx as nx

class Test_FlexaSheet(unittest.TestCase):
    def setUp(self):
        G = nx.Graph()
        a = 1 / np.sqrt(3)
        self.r = np.array([
            [-1 / 2, -a / 2, 0],
            [1 / 2, -a / 2, 0],
            [0, a, 0],
            [0, 0, np.sqrt(2) * a],
            [0, -a, np.sqrt(2) * a],
            [1 / 2, a / 2, np.sqrt(2) * a],
            [-1 / 2, a / 2, np.sqrt(2) * a]
        ])
        
        for i in range(3):
            G.add_node(i, x0=self.r[i, :], cell=True)

        for i in range(3, 7):
            G.add_node(i, x0=self.r[i, :], cell=False)

        self.neigh_collars = {(0, 1): [3, 4], (0, 2): [3, 6], (1, 2): [3, 5]}

        for (k, v) in self.neigh_collars.items():
            G.add_edge(k[0], k[1], collar=False, collar_pts=v)

        self.collar_edges = [(0, 3), (0, 4), (0, 6), (1, 3), (1, 4), (1, 5), 
                             (2, 3), (2, 5), (2, 6)]

        for (i, j) in self.collar_edges:
            G.add_edge(i, j, collar=True)

        self.cell_collars = {0: [3, 4, 6], 1: [3, 4, 5], 2: [3, 5, 6]}

        self.f = FlexaSheet(G)

        ref = np.array([0, 0, 1])
        ab = self.r[3, :] - self.r[0, :]
        self.phi = np.arccos(np.dot(ref, ab / np.linalg.norm(ab)))

        n1 = np.cross(self.r[4, :] - self.r[0, :], self.r[3, :] - self.r[0, :])
        n1 /= np.linalg.norm(n1)
        n2 = n1 * np.array([-1, 1, 1])
        self.psi = (np.pi - np.arccos(np.dot(n1, n2))) / 2

    def assertArrayEqual(self, actual, expected, rtol=1e-05, atol=1e-08):
        self.assertTrue(
            np.all(np.isclose(actual, expected, rtol=rtol, atol=atol)))

    def test_init(self):
        self.assertArrayEqual(self.f.x, self.r.flatten())
        self.assertEqual(self.f.n, self.r.shape[0])
        self.assertEqual(self.f.n_cells, 3)
        
        self.assertEqual(self.f.cell_collars, self.cell_collars)
        
        self.assertEqual(self.f.neigh_collars, self.neigh_collars)

        self.assertEqual(self.f.phi0, self.f.aphi(self.r))
        self.assertEqual(self.f.psi0, self.f.apsi(self.r))

    def test_flatgen(self):
        lattice = np.array([[0, 0], [0, 1], [1, 0]])
        self.assertEqual(FlexaSheet.flatgen(lattice).n_cells, 3)

    def test_cellgraph(self):
        G = nx.Graph()
        G.add_nodes_from([(i, {'x0': self.r[i, :], 'cell': True}) \
            for i in range(3)])
        
        self.assertEqual(FlexaSheet.cellgraph(self.r[:3, :]), G)

    def test_collar_pairs(self):
        collar_pairs = np.array([[3, 4], [3, 6], [3, 5]])
        self.assertArrayEqual(
            FlexaSheet.collar_pairs(self.neigh_collars), collar_pairs)

    def test_cell_degrees(self):
        self.assertEqual(self.f.cell_degrees(), {i: 3 for i in range(3)})

    def test_angle(self):
        self.assertEqual(FlexaSheet.angle([1, 0, 0], [0, 1, 0]), np.pi / 2)

    def test_tri_normal(self):
        ref = np.array([0, 0, 1])
        a = np.array([0, 1])
        b = np.array([1, 0])
        c = np.array([1, 1])
        self.assertArrayEqual(FlexaSheet.tri_normal(a, b, c, ref), ref)

        self.assertArrayEqual(FlexaSheet.tri_normal(np.vstack((a, b, c))), ref)

        a = np.array([0, 1, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 0, 1])
        n = np.array([1, 1, 1]) / np.sqrt(3)
        self.assertArrayEqual(FlexaSheet.tri_normal(a, b, c, ref), n)

        self.assertArrayEqual(
            FlexaSheet.tri_normal(a, b, c, ref), n)
        
        ref = np.array([0, 0, -1])
        self.assertArrayEqual(
            FlexaSheet.tri_normal(np.vstack((a, b, c)), ref=ref), -1 * n)

    def test_face_normal(self):
        n_points = 8 # must be even
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        z = np.where(np.arange(n_points) % 2 == 0, 1, -1)
        r = np.vstack((np.cos(angles), np.sin(angles), z)).T

        ref = np.array([1, 1, 1])
        expected = np.array([0, 0, 1])
        self.assertArrayEqual(FlexaSheet.face_normal(r, ref), expected, 
            atol=1e-10)

        self.assertArrayEqual(FlexaSheet.face_normal(r, -ref), -expected, 
            atol=1e-10)

    def test_cell_normal(self):
        ref = np.array([0, 0, 1])
        self.assertTrue(np.all(self.f.cell_normal(self.r, 0) == ref))

    def test_phis(self):
        phis = self.f.phis(self.r)
        for c in self.cell_collars.keys():
            self.assertArrayEqual(phis[c], self.phi)

    def test_aphi(self):
        self.assertEqual(self.f.aphi(self.r), 
            np.mean(list(self.f.phis(self.r).values())))
    
    def test_phi_energies(self):
        phis = self.f.phis(self.r)
        phi0 = 1
        self.f.phi0 = phi0
        phi_energies = self.f.phi_energies(self.r)
        for c in self.cell_collars.keys():
            self.assertEqual(phi_energies[c], np.sum((phis[c] - phi0) ** 2))
    
    def test_phi_energy(self):
        self.assertAlmostEqual(self.f.phi_energy(self.r), 0, places=10)
        
        self.assertEqual(self.f.phi_energy(self.r), 
            sum(self.f.phi_energies(self.r).values()))

    def test_psis(self):
        psis = self.f.psis(self.r)
        for cells in self.neigh_collars.keys():
            self.assertArrayEqual(psis[cells], self.psi)

    def test_apsi(self):
        n1 = np.cross(self.r[4, :] - self.r[0, :], self.r[3, :] - self.r[0, :])
        n1 /= np.linalg.norm(n1)
        n2 = n1 * np.array([-1, 1, 1])
        self.assertEqual(self.f.apsi(self.r), 
            (np.pi - np.arccos(np.dot(n1, n2))) / 2)

    def test_psi_energies(self):
        psis = self.f.psis(self.r)
        psi0 = 1
        self.f.psi0 = psi0
        psi_energies = self.f.psi_energies(self.r)
        for cells in self.neigh_collars.keys():
            self.assertEqual(psi_energies[cells], 
                np.sum((psis[cells] - psi0) ** 2))

    def test_psi_energy(self):
        self.assertAlmostEqual(self.f.psi_energy(self.r), 0, places=10)
        
        self.assertEqual(self.f.psi_energy(self.r), 
            sum(self.f.psi_energies(self.r).values()))

    def test_collar_lengths(self):
        self.assertTrue(np.all(self.f.collar_lengths(self.r) == 1))

    def test_spring_energy(self):
        self.assertAlmostEqual(self.f.spring_energy(self.r), 0, places=10)
        
        ell0 = 10
        self.f.ell0 = 10
        self.assertEqual(self.f.spring_energy(self.r),
            np.sum((self.f.collar_lengths(self.r) - self.f.ell0) ** 2))
    
    def test_energy(self):
        self.assertEqual(self.f.energy(self.r), 
            self.f.phi_energy(self.r) + self.f.psi_energy(self.r) + \
                self.f.spring_energy(self.r))

    def test_solve_shape(self):
        self.f.solve_shape(1)
        self.assertArrayEqual(self.f.x.reshape((-1, 3)), self.r)
    
if __name__ == '__main__':
    unittest.main()