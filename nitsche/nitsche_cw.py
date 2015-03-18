"""
Demo for Nitsche-type free-slip boundary conditions
"""

__author__    = "Evan Cummings (evan.cummings@umontan.edu"
__copyright__ = "Copyright (c) 2014 %s" % __author__

from fenics import *

mesh = Mesh("meshes/dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, 
                           "meshes/dolfin_fine_subdomains.xml.gz")

# naive equal order element
V = VectorFunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)

W = V*Q

# variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# free-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
u_n = Constant(0.0)

# Inflow boundary condition for velocity
# x0 = 1
u_0 = Expression(("-sin(x[1]*pi)", "0.0"))

# Boundary condition for pressure at outflow
# x0 = 0
p_0 = Constant(0.0)

ds   = Measure("ds")[sub_domains]
dG_0 = ds(0)
dG_r = ds(1)
dG_l = ds(2)

alpha = Constant(1./10)
beta  = Constant(100)
h     = CellSize(mesh)
n     = FacetNormal(mesh)
I     = Identity(2)
eta   = Constant(1.0)
f     = Constant((0.0,0.0))
fric  = Constant(10.0)

def epsilon(u): return 0.5*(grad(u) + grad(u).T)
def sigma(u,p): return 2*eta * epsilon(u) - p*I
def L(u,p):     return -div(sigma(u,p))

B_o = + inner(sigma(u,p), grad(v)) * dx \
      - div(u) * q * dx \
      - alpha * h**2 * inner(L(u,p), L(v,q)) * dx \

B_g = - dot(v,n) * dot(n, dot(sigma(u,p), n)) * dG_0 \
      - dot(u,n) * dot(n, dot(sigma(v,q), n)) * dG_0 \
      - inner(dot(sigma(u,p), n), v) * dG_r \
      - inner(dot(sigma(v,q), n), u) * dG_r \
      + fric**2 * dot(u, v) * dG_0 \
      + beta/h * inner(v,u) * dG_r \
      + beta/h * p * q * dG_l \
      + beta/h * dot(u,n) * dot(v,n) * dG_0 \

F   = + dot(f,v)*dx \
      - alpha * h**2 * inner(f, L(v,q)) * dx \
      - inner(dot(sigma(v,q), n), u_0) * dG_r \
      + beta/h * inner(v,u_0) * dG_r \
      + beta/h * p_0 * q * dG_l \
      + beta/h * u_n * dot(v,n) * dG_0 \

# solve variational problem
wh = Function(W)

solve(B_o + B_g == F, wh,
      solver_parameters = {"linear_solver": "tfqmr", 
                           "preconditioner": "default"} )

uh, ph = wh.split(True)

print "Norm of velocity coefficient vector: %.15g" % uh.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % ph.vector().norm("l2")

File("output/nitsche_cw-velocity.pvd") << uh
File("output/nitsche_cw-pressure.pvd") << ph



