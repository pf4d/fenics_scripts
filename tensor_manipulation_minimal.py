from __future__ import print_function
from dolfin import *
import numpy             as np
import matplotlib.pyplot as plt

# create mesh :
nx,ny,nz = 2,2,2
mesh     = UnitCubeMesh(nx, ny, nz)

# define function space :
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W  = FunctionSpace(mesh, P1)

# form boundary condition over the top face :
def top(x, on_boundary):  return abs(x[2] - 1) < DOLFIN_EPS and on_boundary
m_x_bc   = DirichletBC(W, 11, top)

# create "stiffness" matrix :
A = PETScMatrix(mpi_comm_world())
A.mat().setSizes([W.dim()]*2)
A.mat().setType("aij")
A.mat().setUp()
A.mat().assemble()
A.ident_zeros()

## set items on the diagonal to one :
#for i in range(W.dim()):
#	print(i)
#	block = np.array([1],      dtype = np.float_)
#	rows  = np.array([i],      dtype = np.intc)
#	cols  = np.array([i],      dtype = np.intc)
#	A.set(block,rows,cols)
#
#A.apply('insert')

# define variational problem and assemble the rhs :
v    = TestFunction(W)
x    = Function(W)      # unknown
f    = Constant(1.0)    # source
L    = inner(f, v)*dx
b    = assemble(L)

# set the rhs to something interesting :
b.set_local(np.arange(0, W.dim(), dtype=np.float_))
b.apply('insert')

## get the underlying matricies and vector to operate on :
#A    = as_backend_type(A).mat()
#b    = as_backend_type(b).vec()
#
## convert back to dolfin :
#A    = Matrix(PETScMatrix(A))
#b    = Vector(PETScVector(b))

# plot the resulting matrices :
fig = plt.figure(figsize=(4,4))
ax1 = fig.add_subplot(111)

ax1.imshow(A.array())

plt.tight_layout()
plt.show()

# find where A_n is zero on the diagonal :
zero      = np.array(np.where(A.array() == 0))
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

# this will work! yay!
solve(A, x.vector(), b)

# FIXME: ERROR ERROR ERROR!
# apply boundary conditions to the modified system :
m_x_bc.apply(A)
m_x_bc.apply(b)


