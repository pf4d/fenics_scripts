from dolfin import *
from time   import time

lerr_a = []
herr_a = []
l_t_a  = []
h_t_a  = []
n_a    = [10,20,40,80,160,320,640,1280,2560]

for n in n_a:

  mesh = UnitIntervalMesh(n)

  # Function to project:
  x = SpatialCoordinate(mesh)
  toProject = sin(2.0*pi*x[0])

  # Set up and solve projection problem:
  V  = FunctionSpace(mesh,"Lagrange",1)
  du = TrialFunction(V)
  v  = TestFunction(V)

  delta = + inner(grad(du-toProject),grad(v)) * dx \
          + inner(du-toProject,v) * dx

  u_h1 = Function(V)
  t0_h1 = time()
  solve(lhs(delta) == rhs(delta), u_h1)
  tf_h1 = time()

  t0_l2 = time()
  u_l2 = project(toProject,V)
  t0_l2 = time()
  tf_l2 = time()

  u_exact = interpolate(Expression('sin(2.0*pi*x[0])', degree=2), V)

  # Plot the result:
  herr = norm(u_exact.vector() - u_h1.vector())
  lerr = norm(u_exact.vector() - u_l2.vector())
  print "for n = %i :\t H^1 error = %.4e,\t L^2 error = %.4e " % (n, herr, lerr)

  herr_a.append(herr)
  lerr_a.append(lerr)
  h_t_a.append(tf_h1 - t0_h1)
  l_t_a.append(tf_l2 - t0_l2)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.loglog(n_a, lerr_a, label=r"$L^2$ error")
ax1.loglog(n_a, herr_a, label=r"$H^1$ error")
ax1.legend()
ax1.set_ylabel('norm(projection - exact)')
ax1.set_xlabel('number of dofs')
ax1.grid()

ax2.semilogy(n_a, l_t_a)
ax2.semilogy(n_a, h_t_a)
ax2.set_ylabel('time to compute [s]')
ax2.set_xlabel('number of dofs')
ax2.grid()

plt.tight_layout()
plt.show()


