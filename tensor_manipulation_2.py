from __future__ import print_function
from dolfin import *
import numpy             as np
import matplotlib.pyplot as plt

# create meshes :
nx,ny,nz = 2,2,2
mesh     = UnitCubeMesh(nx, ny, nz)

#===============================================================================
# create a MeshFunction for marking boundaries :
ff   = MeshFunction('size_t', mesh, 2)

# initialize to zero :
ff.set_all(0)

# iterate through the facets and mark each if on a boundary :
#
#   1 =  ..... top           |       4 =  ..... West side
#   2 =  ..... bottom        |       5 =  ..... North side
#   3 =  ..... East side     |       6 =  ..... South side
for f in facets(mesh):
	n       = f.normal()    # unit normal vector to facet f
	if   n.z() >  DOLFIN_EPS and f.exterior():                        ff[f] = 1
	elif n.z() < -DOLFIN_EPS and f.exterior():                        ff[f] = 2
	elif n.x() >  DOLFIN_EPS and n.y() < DOLFIN_EPS and f.exterior(): ff[f] = 3
	elif n.x() < -DOLFIN_EPS and n.y() < DOLFIN_EPS and f.exterior(): ff[f] = 4
	elif n.y() >  DOLFIN_EPS and n.x() < DOLFIN_EPS and f.exterior(): ff[f] = 5
	elif n.y() < -DOLFIN_EPS and n.x() < DOLFIN_EPS and f.exterior(): ff[f] = 6

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W  = FunctionSpace(mesh, P1)

# form boundary condition over the top face :
m_x_bc   = DirichletBC(W, 11, ff, 1)

# get the dofs of just the top face :
b_x_dofs = np.array(m_x_bc.get_boundary_values().keys(), dtype=np.intc)

# sort the dofs (the dict is not sorted) :
b_x_dofs.sort()

# create pseudo-transformation matrix (not really a transformation matrix, but
# this is not related to the problem) :
T = PETScMatrix(mpi_comm_world())
T.mat().setSizes([W.dim()]*2)
T.mat().setType("aij")
T.mat().setUp()
T.mat().assemble()
T.ident_zeros()

PETSc_MAT_NEW_NONZERO_ALLOCATION_ERR = 19
PETSc_FALSE = 0
T.mat().setOption(PETSc_MAT_NEW_NONZERO_ALLOCATION_ERR, PETSc_FALSE)

for i in b_x_dofs:
	print(i)
	block = np.array([1],      dtype = np.float_)
	rows  = np.array([i],      dtype = np.intc)
	cols  = np.array([i],      dtype = np.intc)
	T.set(block,rows,cols)

T.apply('insert')

# define variational problem :
u    = TrialFunction(W)
v    = TestFunction(W)
f    = Constant(0.0)
a    = inner(grad(u), grad(v)) * dx
L    = inner(f, v)*dx

# assemble the stiffness and rhs :
A    = assemble(a)
b    = assemble(L)

# get the underlying matricies and vector to operate on :
T    = as_backend_type(T).mat()
A    = as_backend_type(A).mat()
b    = as_backend_type(b).vec()

# pseudo transform the system of equations TAT^T x = T b :
A_n  = Matrix(PETScMatrix(T))
#A_n  = Matrix(PETScMatrix(T.matMult(A).matTransposeMult(T)))
b_n  = Vector(PETScVector(T * b))

# convert back to dolfin matrix :
A    = Matrix(PETScMatrix(A))
T    = Matrix(PETScMatrix(T))

# plot the resulting matrices :
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.imshow(T.array())
ax2.imshow(A.array())
ax3.imshow(A_n.array())

plt.tight_layout()
plt.show()

# find where A_n is zero on the diagonal :
zero      = np.array(np.where(A_n.array() == 0))
zero_diag = zero[0][np.where(zero[0] == zero[1])[0]]

## get the diagonal of A_n :
#A_n_diag = Vector(mpi_comm_world(), A_n.size(0))
#A_n.get_diagonal(A_n_diag)
#
## set the elements of A_n that are zero on the diagonal to a large number :
#A_n_diag_a = A_n_diag.get_local()
#A_n_diag_a[zero_diag] = 1e8
#A_n_diag.set_local(A_n_diag_a)
#A_n_diag.apply('insert')
#A_n.set_diagonal(A_n_diag)

# apply boundary conditions to the modified system :
m_x_bc.apply(A_n)
m_x_bc.apply(b_n)


