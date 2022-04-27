from flexa.lattice import ico

v, f = ico(3, angle=2*np.pi/5)

s = FlexaSheet.facegen(v, f, phi0=0.654, psi0=0.8375, ell0=1)
s.draw('3d')
plt.show()

s.solve_shape(10)
s.draw('3d')
plt.show()