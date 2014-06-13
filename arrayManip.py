from fenics import *

# Create mesh and define function space
mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u = Expression('1 + x[0]*x[0] + 2*x[1]*x[1] + 4*x[2]*x[2]')
#p  = Expression('x[0] + x[1] + x[2]')
#x  = SpatialCoordinate(mesh)
#
#def u0_boundary(x, on_boundary):
#  return on_boundary
#
#bc = DirichletBC(V, u0, u0_boundary)
#
# Define variational problem
#u = TrialFunction(V)
#v = TestFunction(V)
#f = Constant(-1.0)
#a = p * inner(nabla_grad(u), nabla_grad(v)) * dx
#L = f*v*dx
#
# Compute solution
#u = Function(V)
#solve(a == L, u, bc)

# this works :
u = interpolate(u, V)

# subtract 10 from the solution :
u_n  = project(u - 10.0, V)

# reset the solution :
u_v = u_n.vector().array()
u.vector().set_local(u_v)

File('output/u.pvd') << project(u, V)



