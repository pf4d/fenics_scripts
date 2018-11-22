from dolfin import *
import numpy as np

T     = 2.0        # final time
alpha = 3.0        # parameter alpha
beta  = 1.2        # parameter beta
n_x   = 25
n_y   = 25

# Define boundary condition
u_init = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    degree=2, alpha=alpha, beta=beta, t=0)

u_true = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*x[2]',
                    degree=2, alpha=alpha, beta=beta)

# plot the convergence :
err_a  = []
dof_a  = []
for n_z in [2,4,8,16,32,64,128]:

	# Create mesh and define function space
	mesh = BoxMesh(Point(0,0,0), Point(1,1,T), n_x, n_y, n_z)

	# create a MeshFunction for marking boundaries :
	ff   = MeshFunction('size_t', mesh, 2, 0)

	# iterate through the facets and mark each if on a boundary :
	#
	#   1 =  ..... top
	#   2 =  ..... bottom
	#   3 =  ..... sides
	for f in facets(mesh):
		n       = f.normal()    # unit normal vector to facet f
		tol     = 1e-10
		if   n.z() >=  tol and f.exterior():
			ff[f] = 1
		elif n.z() <= -tol and f.exterior():
			ff[f] = 2
		elif (    (n.x() >  tol  and n.y() < tol) \
		       or (n.x() < -tol  and n.y() < tol) \
		       or (n.y() >  tol  and n.x() < tol) \
		       or (n.y() < -tol  and n.x() < tol)) \
		       and f.exterior():
			ff[f] = 3

	ds = Measure('ds', subdomain_data=ff)

	V  = FunctionSpace(mesh, 'P', 1)

	bc_space = DirichletBC(V, u_true, ff, 3)
	bc_time  = DirichletBC(V, u_init, ff, 2)
	bcs      = [bc_space, bc_time]

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Constant(beta - 2 - 2*alpha)

	# advection operator :
	def Lu(u):  return u.dx(2)

	# mesh facet normal :
	n = FacetNormal(mesh)

	# SUPG intrinsic time parameter :
	tau = CellDiameter(mesh) / 2.0

	a = + u.dx(2) * v * dx \
	    + u.dx(0) * v.dx(0) * dx \
	    + u.dx(1) * v.dx(1) * dx \
	    + tau * Lu(u) * Lu(v) * dx \
	    + u.dx(2) * n[2] * v * ds(1) \

	L = + f*v*dx \
	    + tau * f * Lu(v) * dx \

	# Compute solution
	u = Function(V)
	solve(a == L, u, bcs)

	# Compute error at vertices
	u_e   = interpolate(u_true, V)
	error = norm(u_e.vector() - u.vector(), norm_type='l2')
	print('n_z = %i, \t t = %.2f: error = %.3g' % (n_z, T, error))

	err_a.append(error)
	dof_a.append(V.dim())

import matplotlib.pyplot as plt
import matplotlib        as mpl

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'small'

fig = plt.figure()
ax  = fig.add_subplot(111)

ax.loglog(dof_a, err_a)

ax.set_ylabel(r"$\Vert u - u_{e} \Vert_{\infty}$")
ax.set_xlabel(r"number of dofs")
ax.grid()

plt.tight_layout()
plt.show()

File('error.pvd') << project(u_e - u, V)


