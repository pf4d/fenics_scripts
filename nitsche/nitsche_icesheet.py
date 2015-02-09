"""
Demo for Nitsche-type free-slip boundary conditions
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from fenics import *

l    = 10000
d    = 800
mesh = RectangleMesh(-l+d, 0, l-d, 1, 1000, 10)
ff   = FacetFunction('size_t', mesh, 0)

# Define function spaces
B  = FunctionSpace(mesh, "B", 3)
Q  = FunctionSpace(mesh, "CG", 1)
M  = Q + B
V  = MixedFunctionSpace([M,M])
W  = MixedFunctionSpace([V,Q])

# iterate through the facets and mark each if on a boundary :
#
#   1 = ..... surface
#   2 = ..... base
#   3 = ..... right side
#   4 = ..... left side
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = 1.0
  if   n.y() >=  tol and abs(n.x()) < tol and f.exterior():
    ff[f] = 1
  elif n.y() <= -tol and abs(n.x()) < tol and f.exterior():
    ff[f] = 2
  elif abs(n.y()) < tol and n.x() >= tol and f.exterior():
    ff[f] = 3
  elif abs(n.y()) < tol and n.x() <= -tol and f.exterior():
    ff[f] = 4

class Surface(Expression):
  def eval(self,values,x):
    values[0] = 1000*cos(pi*x[0]/(2*l))
S = Surface(element = Q.ufl_element())

class Bed(Expression):
  def eval(self,values,x):
    values[0] = 100*cos(1000*pi*x[0]/l)
B = Bed(element = Q.ufl_element())

class Depth(Expression):
  def eval(self, values, x):
    values[0] = min(0, x[1]-50)
D = Depth(element = Q.ufl_element())


# Deform the square to the defined geometry :
for x in mesh.coordinates():
  # transform z :
  # thickness = surface - base, z = thickness + base
  x[1]  = x[1] * (S(x[0], x[1]) - B(x[0], x[1]))
  x[1]  = x[1] +  B(x[0], x[1])

File('output/ff.pvd') << ff

# variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
U    = Function(W)

# free-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
u_n = Constant(0.0)
u_0 = Constant((0.0,0.0))

ds     = Measure("ds")[ff]
dSrf   = ds(1)
dBed   = ds(2)
dRight = ds(3)
dLeft  = ds(4)

alpha = Constant(1./10)
beta  = Constant(100)
h     = CellSize(mesh)
n     = FacetNormal(mesh)
x     = SpatialCoordinate(mesh)
I     = Identity(2)
eta   = 1.0
rho   = 917.0
rho_w = 1000.0
g     = 9.8
f     = Constant((0.0, -rho * g))
fric  = Constant(0.1)
f_w   = rho*g*(S - x[1]) + rho_w*g*D

bc0   = DirichletBC(W.sub(0), u_0, ff, 4)
bc1   = DirichletBC(W.sub(0), u_0, ff, 2)
bc    = []

def epsilon(u): return 0.5*(grad(u) + grad(u).T)
def sigma(u,p): return 2*eta * epsilon(u) - p*I
def L(u,p):     return -div(sigma(u,p))

B_o = + inner(sigma(u,p), grad(v)) * dx \
      - div(u) * q * dx \
      - alpha * h**2 * inner(L(u,p), L(v,q)) * dx \

B_g = - dot(v,n) * dot(n, dot(sigma(u,p), n)) * dBed \
      - dot(u,n) * dot(n, dot(sigma(v,q), n)) * dBed \
      + beta/h * dot(u,n) * dot(v,n) * dBed \
      + beta/h * dot(u,n) * dot(v,n) * dLeft \
      + fric**2 * dot(u, v) * dBed \

F   = + dot(f,v)*dx \
      + alpha * h**2 * inner(f, L(v,q)) * dx \
      - u_n * dot(n, dot(sigma(v,q), n)) * dBed \
      - u_n * dot(n, dot(sigma(v,q), n)) * dLeft \
      + beta/h * u_n * dot(v,n) * dBed \
      + f_w * dot(v,n) * dRight \

solve(B_o + B_g == F, U, bc)

uh, ph = U.split(True)

print "Norm of velocity coefficient vector: %.15g" % uh.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % ph.vector().norm("l2")

File("output/nitsche_velocity.pvd") << uh
File("output/nitsche_pressure.pvd") << ph



