from dolfin import *

# Scaled variables
l       = 1
mu      = 1
rho     = 1
gamma   = 0.4
beta    = 1.25
lambda_ = beta
g       = gamma

# define boundary condition :
def clamped_boundary(x, on_boundary):
  return on_boundary and x[0] < 1e-14

# define strain :
def epsilon(u):   return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# define stress :
def sigma(u):     return lambda_*nabla_div(u)*Identity(3) + 2*mu*epsilon(u)

for n in [5,10,25]:

  # Create mesh and define function space
  mesh    = UnitCubeMesh(n, n, 2)
  bmesh   = BoundaryMesh(mesh, 'exterior')
  cellmap = bmesh.entity_map(2)
  pb      = MeshFunction("size_t", bmesh, 2, 0)
  for c in cells(bmesh):
    if Facet(mesh, cellmap[c.index()]).normal().z() < -1e-3:
      pb[c] = 1
  mesh = SubMesh(bmesh, pb, 1)

  Qe   = FiniteElement('CG', mesh.ufl_cell(), 1)
  Ve   = MixedElement([Qe]*3)
  Q    = FunctionSpace(mesh, Qe)
  V    = FunctionSpace(mesh, Ve)

  bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Constant((0, 0, -rho*g))
  T = Constant((0, 0, 0))
  a = inner(sigma(u), epsilon(v))*dx
  L = dot(f, v)*dx + dot(T, v)*ds

  # Compute solution
  u = Function(V)
  solve(a == L, u, bc)

  # the magnitude of displacement is in this case only in the z direction :
  u_mag = u.split(True)[2]

  # get varaibles for Nanson's formula :
  k     = Constant((0,0,1))
  N     = k
  F     = Identity(3) + grad(u)
  J     = det(F)
  n_mag = sqrt(u_mag.dx(0)**2 + u_mag.dx(1)**2 + 1)
  n     = (k - nabla_grad(u_mag)) / n_mag

  A_volume    = assemble( J * dx )
  A_nansen    = assemble( J * dot(dot(inv(F.T), N), n) * dx )
  A_geometric = assemble( n_mag * dx )

  print 'error "volume" = %.2e\t error "geometric" = %.2e' \
        % (abs(A_volume - A_nansen), abs(A_geometric - A_nansen))

import matplotlib.pyplot as plt
ax = plot(u_mag)
plt.show()


