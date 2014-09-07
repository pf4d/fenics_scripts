"""
Demo for Nitsche-type free-slip boundary conditions
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from fenics import *

# solver parameters :
parameters['form_compiler']['quadrature_degree'] = 3
params = {"newton_solver":
         {"linear_solver"        : 'mumps',
          "preconditioner"       : 'default',
          "maximum_iterations"   : 35,
          "relaxation_parameter" : 0.7,
          "relative_tolerance"   : 1e-3,
          "absolute_tolerance"   : 1e-3}}

#nx      = 20
#ny      = 20
#nz      = 10
#mesh    = BoxMesh(-1,-1, 0, 1, 1, 1, nx,ny,nz)
mesh    = Mesh('../meshes/unit_cyl_mesh.xml')

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
    values[2] = 0.0
    values[2] = -100*gauss(x[0], x[1], t/2, t/2)
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

bcs = [bc0]

g      = 9.81
rho    = 917.0
gamma  = 8.71e-4
E      = 1.0
R      = 8.31447
N      = 3.0
theta  = 273.15
x      = SpatialCoordinate(mesh)
alpha  = Constant(1.0/10.0)
beta   = Constant(100000)
h      = CellSize(mesh)
n      = FacetNormal(mesh)
I      = Identity(3)
fric   = Constant(0.0)
i      = as_vector([1,0,0])
j      = as_vector([0,1,0])
k      = as_vector([0,0,1])
f      = -rho*g*j
f_w    = Expression("rho*g*(sqrt(pow(2*t,2) - pow(x[0],2)) - x[1])", 
                    rho=rho, g=g, t=t, element=Q.ufl_element())

# rate-factor :
Tstar = theta + gamma * (S - x[2])
a_T   = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10)
Q_T   = conditional( lt(Tstar, 263.15), 6e4,          13.9e4)
A     = E * a_T * exp( -Q_T / (R * Tstar))
#A     = 1e-8
b     = A**(-1/N)

def epsilon(u): return 0.5*(grad(u) + grad(u).T)
def sigma(u,p): return 2*eta*epsilon(u) - p*I
def L(u,p):     return -div(sigma(u,p))

# Second invariant of the strain rate tensor squared
def epsdot(U):
  """
  return the 2nd invariant of the strain-rate tensor of <U>, squared.
  """
  u,v,w = U
  epi   = epsilon(U)
  #epi02 = 0.5*u.dx(2)
  #epi12 = 0.5*v.dx(2)
  #epi   = as_matrix([[epi[0,0],  epi[0,1],  epi02   ],
  #                   [epi[1,0],  epi[1,1],  epi12   ],
  #                   [epi02,     epi12,     epi[2,2]]])
  ep_xx  = epi[0,0]
  ep_yy  = epi[1,1]
  ep_xy  = epi[0,1]
  ep_xz  = epi[0,2]
  ep_yz  = epi[1,2]
  epsdot = ep_xx**2 + ep_yy**2 + ep_xx*ep_yy + ep_xy**2 + ep_xz**2 + ep_yz**2
  return epsdot

# viscosity :
eta = 0.5 * b * (epsdot(u) + DOLFIN_EPS)**((1-N)/(2*N))
#eta = Constant(10000000)

B_o = + inner(sigma(u,p), grad(v)) * dx \
      - div(u) * q * dx \
      - alpha * h**2 * inner(L(u,p), L(v,q)) * dx \

B_g = - dot(v,n) * dot(n, dot(sigma(u,p), n)) * dGamma \
      - dot(u,n) * dot(n, dot(sigma(v,q), n)) * dGamma \
      - fric**2 * dot(u, v) * dGamma \
      + beta/h * dot(u,n) * dot(v,n) * dGamma \
      - f_w * dot(v, n) * (dBed + dSrf) \
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

solve(R == 0, U, bcs, J=J, solver_parameters=params)

uh, ph = U.split(True)

print "Norm of velocity coefficient vector: %.15g" % uh.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % ph.vector().norm("l2")

File("output/nitsche_cw_newton_3d-velocity.pvd") << uh
File("output/nitsche_cw_newton_3d-pressure.pvd") << ph
File("output/u_0.pvd") << interpolate(u_0,V)
File('output/f_w.pvd') << interpolate(f_w, Q)



