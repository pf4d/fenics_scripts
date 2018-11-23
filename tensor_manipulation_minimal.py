from __future__ import print_function
from dolfin import *
import numpy             as np
import matplotlib.pyplot as plt

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

# create mesh :
nx,ny,nz = 2,2,2
mesh     = UnitCubeMesh(nx, ny, nz)

# define function space :
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W  = FunctionSpace(mesh, P1)

# define variational problem and assemble the rhs :
u    = TrialFunction(W)
v    = TestFunction(W)
x    = Function(W)      # unknown
f    = Constant(1.0)    # source
L    = inner(f, v)*dx
b    = assemble(L)

# form boundary condition over the top face :
def top(x, on_boundary):  return abs(x[2] - 1) < DOLFIN_EPS and on_boundary
m_x_bc   = DirichletBC(W, 11, top)

# Create a matrix full of zeros by integrating over empty domain.  (No elements
# are flagged with ID 1.)  For some reason this seems to be the only way I can
# find to create A that allows BCs to be set without errors:
A  = assemble(inner(u,v)*dx(1))

# convert to the underlying representation :
A  = as_backend_type(A).mat()
b  = as_backend_type(b).vec()

# If you know how many nonzeros per row are needed, you can do something like
# the following:
nonzerosPerRow = 1
A.setPreallocationNNZ([nonzerosPerRow,nonzerosPerRow])
A.setUp()

# The following can be uncommented for this code to work even if you don't
# know how many nonzeros per row to allocate:
#A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

# Now edit the "blank" matrix manually (set to identity matrix for testing) :
Istart, Iend = A.getOwnershipRange()
for I in range(Istart,Iend): A[I,I] = 1.0
A.assemble()

# manipulate both sides :
A_n  = Matrix(PETScMatrix(A.transposeMatMult(A)))
b_n  = Vector(PETScVector(A * b))

# this will work! yay!
solve(A_n, x.vector(), b_n)

# verify that the edited A acts as the identity matrix, as desired :
print(norm(x.vector()-b_n))

# this will work! yay!
m_x_bc.apply(b_n)

# FIXME: ERROR ERROR ERROR!
# apply boundary conditions to the modified system :
m_x_bc.apply(A_n)


