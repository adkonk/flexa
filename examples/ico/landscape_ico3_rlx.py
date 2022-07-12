from landscape_utils import phis, psis, relax_sheet, save_dir, ell0

name = 'ico3'
k_orig = (1, 2, 5)
k_temp = (1, 2, 0.1)

pi = 0
n_sheets = energies[name].size
for (phi_i, psi_i) in product(range(phis.size), range(psis.size)):
    print('\nSheet %d out of %d with phi0=%0.2f, psi0=%0.2f' % \
        (pi + 1, n_sheets, phis[phi_i], psis[psi_i]))
    pi += 1
    s = relax_sheet(name, phis[phi_i], psis[psi_i], ell0, k_orig, k_temp, 
        save_dir)
    energies[psi_i, phi_i] = s.energy(s.x, k_orig)
    np.save(os.path.join(save_dir, 'energies_%s_rlx.npy' % name)

