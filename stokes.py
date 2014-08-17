from dolfin import *
import numpy as np

def normalize_vector(U):
  """
  Create a normalized vector of the UFL vector <U>.
  """
  # iterate through each component and convert to array :
  U_v = []
  for u in U:
    # convert to array and normailze the components of U :
    u_v = u.vector().array()
    U_v.append(u_v)
  U_v = np.array(U_v)

  # calculate the norm :
  norm_u = np.sqrt(sum(U_v**2))
  
  # normalize the vector :
  U_v /= norm_u
  
  # convert back to fenics :
  U_f = []
  for u_v in U_v:
    u_f = Function(Q)
    u_f.vector().set_local(u_v)
    u_f.vector().apply('insert')
    U_f.append(u_f)

  # return a UFL vector :
  return as_vector(U_f)

def vert_integrate(u, Q, ff):
  """
  Integrate <u> from the bed to the surface.
  """
  phi    = TestFunction(Q)               # test function
  v      = TrialFunction(Q)              # trial function
  bc     = DirichletBC(Q, 0.0, ff, 2)    # integral is zero on bed (ff = 2) 
  a      = v.dx(2) * phi * dx            # rhs
  L      = u * phi * dx                # lhs
  v      = Function(Q)                   # solution function
  solve(a == L, v, bc)                   # solve
  return v

def extrude(f, b, d, Q, ff):
  """
  This extrudes a function <f> defined along a boundary <b> out onto
  the domain in the direction <d>.  It does this by formulating a 
  variational problem:
  """
  phi = TestFunction(Q)
  v   = TrialFunction(Q)
  a   = v.dx(d) * phi * dx
  L   = DOLFIN_EPS * phi * dx
  bc  = DirichletBC(Q, f, ff, b)
  v   = Function(Q)
  solve(a == L, v, bc)
  return v

def strain_rate(U):
  """
  return the strain-rate tensor of <U>.
  """
  epsdot  = 0.5 * (grad(U) + grad(U).T)
  return epsdot
 
nx      = 40
ny      = 40
nz      = 5
mesh    = UnitCubeMesh(nx,ny,nz)

# Define function spaces
Q  = FunctionSpace(mesh, "CG", 1)
V  = VectorFunctionSpace(mesh, "CG", 2)
W  = V * Q
ff = FacetFunction('size_t', mesh, 0)

# iterate through the facets and mark each if on a boundary :
#
#   1 = high slope, upward facing ................ surface
#   2 = high slope, downward facing .............. base
#   3 = low slope, upward or downward facing ..... east side
#   4 = low slope, upward or downward facing ..... west side
#   5 = low slope, upward or downward facing ..... north side
#   6 = low slope, upward or downward facing ..... south side
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = 1e-3
  if   n.z() >=  tol and f.exterior():
    ff[f] = 1
  elif n.z() <= -tol and f.exterior():
    ff[f] = 2
  elif n.z() >  -tol and n.z() < tol  and f.exterior() \
                     and n.x() > tol  and n.y() < tol :
    ff[f] = 3
  elif n.z() >  -tol and n.z() < tol  and f.exterior() \
                     and n.x() < -tol and n.y() < tol :
    ff[f] = 4
  elif n.z() >  -tol and n.z() < tol  and f.exterior() \
                     and n.x() < tol  and n.y() > tol :
    ff[f] = 5
  elif n.z() >  -tol and n.z() < tol  and f.exterior() \
                     and n.x() < tol  and n.y() < -tol :
    ff[f] = 6

ds     = Measure('ds')[ff]
dSrf   = ds(1)
dBed   = ds(2)
dEst   = ds(3)
dWst   = ds(4)
dNth   = ds(5)
dSth   = ds(6)
dGamma = dEst + dWst + dNth + dSth

alpha   = 1.0 * pi / 180
L       = 40000.0
S0      = 1000.0
bm      = 200.0

class Surface(Expression):
  def eval(self,values,x):
    values[0] = S0 + x[0] * tan(alpha)
S = Surface(element = Q.ufl_element())

class Bed(Expression):
  def eval(self,values,x):
    values[0] = + S0 + x[0] * tan(alpha) \
                - 1000.0 \
                + 500.0 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)
B = Bed(element = Q.ufl_element())

class Depth(Expression):
  def eval(self, values, x):
    values[0] = min(0, x[2])
D = Depth(element = Q.ufl_element())

class Beta(Expression):
  def eval(self, values, x):
    values[0] = bm - bm * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)
beta = Beta(element = Q.ufl_element())

xmin = -L
xmax = 0
ymin = -L
ymax = 0

# width and origin of the domain for deforming x coord :
width_x  = xmax - xmin
offset_x = xmin

# width and origin of the domain for deforming y coord :
width_y  = ymax - ymin
offset_y = ymin

# Deform the square to the defined geometry :
for x in mesh.coordinates():
  # transform x :
  x[0]  = x[0]  * width_x + offset_x

  # transform y :
  x[1]  = x[1]  * width_y + offset_y

  # transform z :
  # thickness = surface - base, z = thickness + base
  x[2]  = x[2] * (S(x[0], x[1], x[2]) - B(x[0], x[1], x[2]))
  x[2]  = x[2] +  B(x[0], x[1], x[2])

# constants :
rho    = 917.0
rho_w  = 1000.0
g      = 9.81
gamma  = 8.71e-4
E      = 1.0
R      = 8.314
n      = 3.0
T      = 250.0
x      = SpatialCoordinate(mesh)
N      = FacetNormal(mesh)
h      = CellSize(mesh)
I      = Identity(3)

# solver parameters :
#parameters['form_compiler']['quadrature_degree'] = 2
params = {"newton_solver":
         {"maximum_iterations"   : 25,
          "relaxation_parameter" : 0.8,
          "relative_tolerance"   : 1e-4,
          "absolute_tolerance"   : 1e-16}}

# create functions for boundary conditions :
noslip = Constant((0, 0, 0))
inflow = Expression(("200*sin(x[1]*pi/L)", "0", "0"), L=L)
f_w    = rho*g*(S - x[2]) + rho_w*g*D

# boundary condition for velocity :
bc1 = DirichletBC(V, noslip, ff, 5)
bc2 = DirichletBC(V, noslip, ff, 6)
bc3 = DirichletBC(V, inflow, ff, 3)
bcs = [bc1, bc2, bc3]
bcs = []

#===============================================================================
# define variational problem :
F   = Function(W)
dU  = TrialFunction(W)
Phi = TestFunction(V)
xi  = TestFunction(Q)

U,   P        = split(F)
phi, psi, chi = split(Phi)

# rate-factor :
Tstar = T + gamma * (S - x[2])
a_T   = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10)
Q_T   = conditional( lt(Tstar, 263.15), 6e4,          13.9e4)
A     = E * a_T * exp( -Q_T / (R * Tstar))
b     = A**(-1/n)

# Second invariant of the strain rate tensor squared
epi   = strain_rate(U)
ep_xx = epi[0,0]
ep_yy = epi[1,1]
ep_xy = epi[0,1]
ep_xz = epi[0,2]
ep_yz = epi[1,2]

epsdot = ep_xx**2 + ep_yy**2 + ep_xx*ep_yy + ep_xy**2 + ep_xz**2 + ep_yz**2
eta    = 0.5 * b * (epsdot + 1e-10)**((1-n)/(2*n))

sigma   = 2*eta*epi + P*I

R = + dot(sigma[0], grad(phi)) * dx \
    + dot(sigma[1], grad(psi)) * dx \
    + dot(sigma[2], grad(chi)) * dx \
    + rho * g * Phi * dx \
    + beta**2 * dot(U, Phi) * dBed \
    - f_w * dot(N, Phi) * dGamma \
    + xi * div(U) * dx \

# Jacobian :
J = derivative(R, U, dU)

# compute solution :
solve(R == 0, F, bcs, J=J, solver_parameters=params)

File("output/U.pvd")    << U



