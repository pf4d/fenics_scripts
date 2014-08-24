from dolfin import *

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V    = FunctionSpace(mesh, "Lagrange", 1)
ff   = FacetFunction('size_t', mesh, 0)

# iterate through the facets and mark each if on a boundary :
#
# 1 - West
# 2 - East
# 3 - North
# 4 - South
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = DOLFIN_EPS
  if   n.x() <= -tol and n.y() <   tol and f.exterior():
    ff[f] = 1
  elif n.x() >=  tol and n.y() <   tol and f.exterior():
    ff[f] = 2
  elif n.x() <   tol and n.y() >=  tol and f.exterior():
    ff[f] = 3
  elif n.x() <   tol and n.y() <= -tol and f.exterior():
    ff[f] = 4

ds = Measure('ds')[ff]
dN = ds(3)
dS = ds(4)
dE = ds(2)
dW = ds(1)

dGamma_n = dN + dS
dGamma_d = dE + dW

# Define boundary condition
u0 = Constant(1.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
N = FacetNormal(mesh)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")

a = dot(grad(v), N) * dot(grad(u), N) * ds
b = inner(grad(v), grad(u)) * dx

A = PETScMatrix()
B = PETScMatrix()

A = assemble(a, tensor=A)
B = assemble(b, tensor=B)

eigensolver = SLEPcEigenSolver(A,B)
eigensolver.solve()
C = eigensolver.get_eigenvalue()[0]

beta = 2*C**2 + 1e-1

a = + inner(grad(u), grad(v))*dx \
    - v*dot(grad(u), N)*dGamma_d \
    - u*dot(grad(v), N)*dGamma_d \
    + beta*v*u*dGamma_d
L = + f*v*dx \
    + g*v*dGamma_n \
    - u0*dot(grad(v), N)*dGamma_d \
    + beta*v*u0*dGamma_d

# Compute solution
u = Function(V)
solve(a == L, u)

# Save solution in VTK format
File("poisson.pvd") << u

