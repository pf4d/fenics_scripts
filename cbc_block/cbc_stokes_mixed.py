"""This demo solves the steady Stokes equations for a lid driven
cavity.  The variational problem is formulated using dolfin
mixed-spaces, but the form is assembled in blocks instead of one
monolithic matrix.

"""
from dolfin import *
from block import *
from block.algebraic.petsc import *
from block.iterative import *
from time import time

t0 = time()

mesh = UnitSquareMesh(32, 32)

P2 = VectorElement("Lagrange", triangle, 2)
P1 = FiniteElement("Lagrange", triangle, 1)
TH = MixedElement([P2, P1])

W = FunctionSpace(mesh, TH)

f = Constant((0., 0.))

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

a = inner(grad(u), grad(v)) * dx \
  + p * div(v) * dx \
  + q * div(u) * dx

b = inner(grad(u), grad(v)) * dx + inner(u, v) * dx \
  + p * q * dx


L = inner(f, v) * dx

bcs = [DirichletBC(W.sub(0), (0., 0.), "on_boundary&&(x[1]<1-DOLFIN_EPS)"),
       DirichletBC(W.sub(0), (1., 0.), "on_boundary&&(x[1]>1-DOLFIN_EPS)")]

# assemble as block matrices
A, y = block_assemble(a, L, bcs)
B, _ = block_assemble(b, L, bcs)


# build the preconditioner
P = block_mat([[AMG(B[0,0]),           0],
               [          0, SOR(B[1,1])]])

# We don't want to solve too precisely since we have not accounted 
# for the constant pressure nullspace 
Ainv = MinRes(A, precond = P, relativeconv = True, tolerance = 1e-5, show = 3)
x = Ainv * y

# plotting
V, Q = [sub_space.collapse() for sub_space in W.split()]
u, p = map(Function, [V, Q], x)
plot(u)
plot(p)
interactive()
