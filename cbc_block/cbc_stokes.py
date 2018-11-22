"""This demo solves the Stokes equations with block preconditioning.

The original demo is found in demo/undocumented/stokes-taylor-hood/python.

The algebraic system to be solved can be written as

  BB^ AA [sigma u]^T = BB^ [b 0]^T,

where AA is a 2x2 block system with zero in the (2,2) block

       | A   B |
  AA = |       |,
       | C   0 |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |.
       | 0   L |
"""

from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *
from time import time

import os

t0 = time()

#dolfin.set_log_level(13)

# Load mesh and subdomains
mesh        = Mesh("./dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "./dolfin_fine_subdomains.xml.gz")
dim         = mesh.topology().dim()

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(V, noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(Q, zero, sub_domains, 2)

# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))

a11 = inner(grad(v), grad(u))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx

I  = assemble(p*q*dx)

# Create the block matrix/vector, and apply boundary conditions. A diagonal
# matrix is automatically created to replace the (2,2) block in AA, since bc2
# makes the block non-zero.
bcs = [[bc0, bc1], bc2]
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])
bb  = block_assemble([L1, 0])

block_bc(bcs, True).apply(AA).apply(bb)

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create preconditioners: An ML preconditioner for A, and the inverse diagonal
# of the mass matrix for the (2,2) block.
Ap = ML(A, nullspace=rigid_body_modes(V))
Ip = LumpedInvDiag(I)

prec = block_mat([[Ap, 0],
                  [0, Ip]])

# Create the block inverse, using the preconditioned Minimum Residual method
# (suitable for symmetric indefinite problems).
AAinv = MinRes(AA, precond=prec, tolerance=1e-10, maxiter=500, show=2)

# Compute solution
t1 = time()
u, p = AAinv * bb
tf = time()

print("time to compute: %f" % (tf - t0))
print("time to solve:   %f" % (tf - t1))

print "Norm of velocity coefficient vector: %.15g" % u.norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.norm("l2")

# Plot solution
File('u.pvd') << Function(V, u)

