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

# Define boundary condition
u0  = Constant(1.0)
bcE = DirichletBC(V, u0, ff, 2)
bcW = DirichletBC(V, u0, ff, 1)

bc = [bcE, bcW]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*(dN + dS)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
File("poisson.pvd") << u

