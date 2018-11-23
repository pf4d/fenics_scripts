from __future__ import print_function
from dolfin import *
import numpy             as np
import matplotlib.pyplot as plt

# create meshes :
nx,ny,nz = 1,1,1
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
W  = FunctionSpace(mesh, TH)

# form boundary condition over the top face :
m_x_bc   = DirichletBC(W.sub(0).sub(0), 11, ff, 1)
m_y_bc   = DirichletBC(W.sub(0).sub(1), 22, ff, 1)
m_z_bc   = DirichletBC(W.sub(0).sub(2), 33, ff, 1)

# get the dofs of just the top face :
b_x_dofs = np.array(m_x_bc.get_boundary_values().keys(), dtype=np.intc)
b_y_dofs = np.array(m_y_bc.get_boundary_values().keys(), dtype=np.intc)
b_z_dofs = np.array(m_z_bc.get_boundary_values().keys(), dtype=np.intc)

# sort the dofs (the dict is not sorted) :
b_x_dofs.sort()
b_y_dofs.sort()
b_z_dofs.sort()

# define variational problem :
U    = TrialFunction(W)
V    = TestFunction(W)
u, p = split(U)
v, q = split(V)
f    = Constant((0, 0, 0))
a    = inner(grad(u), grad(v)) * dx
L    = inner(f, v)*dx

# assemble the stiffness and rhs :
A    = assemble(a)
b    = assemble(L)

# Create a matrix full of zeros by integrating over empty domain.  (No elements
# are flagged with ID 1.)  For some reason this seems to be the only way I can
# find to create A that allows BCs to be set without errors:
T = assemble(inner(U,V)*dx(999), keep_diagonal=True)
T.ident_zeros()

# get the underlying matricies and vector to operate on :
T    = as_backend_type(T).mat()
A    = as_backend_type(A).mat()
b    = as_backend_type(b).vec()

# If you know how many nonzeros per row are needed, you can do something like
# the following:
nonzerosPerRow = 3
T.setPreallocationNNZ([nonzerosPerRow,nonzerosPerRow])
T.setUp()

# The following can be uncommented for this code to work even if you don't
# know how many nonzeros per row to allocate:
#T.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

# set to identity matrix first :
Istart, Iend = T.getOwnershipRange()
for i in range(Istart, Iend): T[i,i] = 1.0

# then set the valuse of the transformation tensor :
for i,j,k in zip(b_x_dofs, b_y_dofs, b_z_dofs):
	print(i,j,k)
	T[i,i] = 1
	T[i,j] = 2
	T[i,k] = 3
	T[j,i] = 4
	T[j,j] = 5
	T[j,k] = 6
	T[k,i] = 7
	T[k,j] = 8
	T[k,k] = 9
T.assemble()

# pseudo transform the system of equations TAT^T x = T b :
A_n  = Matrix(PETScMatrix(T.matMult(A).matTransposeMult(T)))
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

# get the diagonal of A_n :
A_n_diag = Vector(mpi_comm_world(), A_n.size(0))
A_n.get_diagonal(A_n_diag)

# set the elements of A_n that are zero on the diagonal to a large number :
A_n_diag_a = A_n_diag.get_local()
A_n_diag_a[zero_diag] = 1e8
A_n_diag.set_local(A_n_diag_a)
A_n_diag.apply('insert')
A_n.set_diagonal(A_n_diag)

# apply boundary conditions to the modified system :
m_x_bc.apply(b_n)

# NOTE: this fails with :
#
#RuntimeError:
#
#*** -------------------------------------------------------------------------
#*** DOLFIN encountered an error. If you are not able to resolve this issue
#*** using the information listed below, you can ask for help at
#***
#***     fenics-support@googlegroups.com
#***
#*** Remember to include the error message listed below and, if possible,
#*** include a *minimal* running example to reproduce the error.
#***
#*** -------------------------------------------------------------------------
#*** Error:   Unable to set given (local) rows to identity matrix.
#*** Reason:  some diagonal elements not preallocated (try assembler
#***          option keep_diagonal).
#*** Where:   This error was encountered inside PETScMatrix.cpp.
#*** Process: 0
#***
#*** DOLFIN version: 2017.2.0
#*** Git changeset:  4c59bbdb45b95db2f07f4e3fd8985c098615527f
#*** -------------------------------------------------------------------------
#
m_x_bc.apply(A_n)


