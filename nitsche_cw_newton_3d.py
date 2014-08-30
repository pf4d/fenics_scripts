"""
Demo for Nitsche-type free-slip boundary conditions
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from fenics import *

parameters['form_compiler']['quadrature_degree'] = 2

nx      = 20
ny      = 20
nz      = 10
#mesh    = BoxMesh(-1,-1, 0, 1, 1, 1, nx,ny,nz)
mesh    = Mesh('meshes/unit_cyl_mesh_crude.xml')

# Define function spaces
#Q  = FunctionSpace(mesh, "CG", 1)
#V  = VectorFunctionSpace(mesh, "CG", 2)
#W  = V * Q
V  = VectorFunctionSpace(mesh, "CG", 1)
B  = VectorFunctionSpace(mesh, "B", 4)
Q  = FunctionSpace(mesh, "CG", 1)
W  = (V + B)*Q
ff = FacetFunction('size_t', mesh, 0)

# iterate through the facets and mark each if on a boundary :
#
#   1 = high slope, upward facing ................ surface
#   2 = high slope, downward facing .............. base
#   3 = low slope, upward or downward facing ..... side
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = 1.0
  if   n.z() >=  tol and f.exterior():
    ff[f] = 1
  elif n.z() <= -tol and f.exterior():
    ff[f] = 2
  elif abs(n.z()) < tol and f.exterior():
    ff[f] = 3

ds      = Measure('ds')[ff]
dSrf    = ds(1)
dBed    = ds(2)
dGamma  = ds(3)

t       = 100.0 / 2
S0      = 0.0
bm      = 100.0

def gauss(x, y, sigx, sigy):
  return exp(-((x/(2*sigx))**2 + (y/(2*sigy))**2))

class Surface(Expression):
  def eval(self,values,x):
    values[0] = S0
S = Surface(element = Q.ufl_element())

class Bed(Expression):
  def eval(self,values,x):
    values[0] = + S0 - 200.0
B = Bed(element = Q.ufl_element())

class U_0(Expression):
  def eval(self,values,x):
    values[0] = 0.0
    values[1] = 0.0
    values[2] = -1
    values[2] = -10*gauss(x[0], x[1], t/2, t/2)
u_0 = U_0(element = V.ufl_element())

xmin = -t
xmax = t
ymin = -t
ymax = t

# width and origin of the domain for deforming x coord :
width_x  = xmax - xmin
offset_x = xmin

# width and origin of the domain for deforming y coord :
width_y  = ymax - ymin
offset_y = ymin

# Deform the square to the defined geometry :
for x in mesh.coordinates():
  # transform x :
  x[0]  = x[0]  * width_x

  # transform y :
  x[1]  = x[1]  * width_y

  # transform z :
  # thickness = surface - base, z = thickness + base
  x[2]  = x[2] * (S(x[0], x[1], x[2]) - B(x[0], x[1], x[2]))
  x[2]  = x[2] +  B(x[0], x[1], x[2])

# variational problem
U    = Function(W)
u, p = split(U)
Phi  = TestFunction(W)
v, q = split(Phi)
dU   = TrialFunction(W)

# free-slip boundary condition for velocity 
u_n = Constant(0.0)

# Inflow boundary condition for velocity
#u_0 = Expression(("0.0", "0.0", "-sin(x[0]*pi/t)*sin(x[1]*pi/t)"), t=t)
bc0 = DirichletBC(W.sub(0), u_0, ff, 1)
bc2 = DirichletBC(W.sub(0), u_0, ff, 2)

# Boundary condition for pressure at outflow
p_0 = Constant(0.0)
bc1 = DirichletBC(W.sub(1), p_0, ff, 2)

bcs = [bc0, bc2]

alpha = Constant(1.0/10.0)
beta  = Constant(100)
h     = CellSize(mesh)
n     = FacetNormal(mesh)
I     = Identity(3)
eta   = Constant(1.0)
f     = Constant((0.0,0.0,0.0))
fric  = Constant(0.0)

def epsilon(u): return 0.5*(grad(u) + grad(u).T)
def sigma(u,p): return 2*eta*epsilon(u) - p*I
def L(u,p):     return -div(sigma(u,p))

B_o = + inner(sigma(u,p), grad(v)) * dx \
      - div(u) * q * dx \
      - alpha * h**2 * inner(L(u,p), L(v,q)) * dx \

B_g = - dot(v,n) * dot(n, dot(sigma(u,p), n)) * dGamma \
      - dot(u,n) * dot(n, dot(sigma(v,q), n)) * dGamma \
      + fric**2 * dot(u, v) * dGamma \
      + beta/h * dot(u,n) * dot(v,n) * dGamma \
#      + beta/h * inner(v,u) * dSrf \
#      + beta/h * p * q * dBed \
#      - inner(dot(sigma(u,p), n), v) * dSrf \
#      - inner(dot(sigma(v,q), n), u) * dSrf \

F   = + dot(f,v)*dx \
      + alpha * h**2 * inner(f, L(v,q)) * dx \
      - u_n * dot(n, dot(sigma(v,q), n)) * dGamma \
      + beta/h * u_n * dot(v,n) * dGamma \
#      - inner(dot(sigma(v,q), n), u_0) * dSrf \
#      + beta/h * inner(v,u_0) * dSrf \
#      + beta/h * p_0 * q * dBed \

R = B_o + B_g - F

J = derivative(R, U, dU)

File("output/u_0.pvd") << interpolate(u_0,V)

solve(R == 0, U, bcs, J=J)

uh, ph = U.split(True)

print "Norm of velocity coefficient vector: %.15g" % uh.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % ph.vector().norm("l2")

File("output/nitsche_cw_newton_3d-velocity.pvd") << uh
File("output/nitsche_cw_newton_3d-pressure.pvd") << ph
File("output/u_0.pvd") << interpolate(u_0,V)



