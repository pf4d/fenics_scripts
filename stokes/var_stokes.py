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
  u,v,w = U
  epi   = 0.5 * (grad(U) + grad(U).T)
  epi02 = 0.5*u.dx(2)
  epi12 = 0.5*v.dx(2)
  epsdot = as_matrix([[epi[0,0],  epi[0,1],  epi02   ],
                      [epi[1,0],  epi[1,1],  epi12   ],
                      [epi02,     epi12,     epi[2,2]]])
  return epsdot
 
top     = Point(0.0, 0.0, 1.0)
bot     = Point(0.0, 0.0, 0.0)
cone    = Cone(top, bot, 1.0, 1.0)
#mesh    = Mesh(cone,20)
nx      = 20
ny      = 20
nz      = 10
#mesh    = BoxMesh(-1,-1, 0, 1, 1, 1, nx,ny,nz)
mesh    = Mesh('meshes/unit_cyl_mesh.xml')

# Define function spaces
Q  = FunctionSpace(mesh, "CG", 1)
V  = VectorFunctionSpace(mesh, "CG", 2)
W  = V * Q
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

ds     = Measure('ds')[ff]
dSrf   = ds(1)
dBed   = ds(2)
dGamma = ds(3)

L       = 40000.0 / 2
S0      = 50.0
bm      = 100.0

def gauss(x, y, sigx, sigy):
  return exp(-((x/(2*sigx))**2 + (y/(2*sigy))**2))

class Surface(Expression):
  def eval(self,values,x):
    values[0] = S0 + 800*gauss(x[0], x[1], L/2, L/2)
S = Surface(element = Q.ufl_element())

class Bed(Expression):
  def eval(self,values,x):
    values[0] = + S0 - 200.0 \
                - 200.0 * gauss(x[0], x[1], L/2, L/2)
B = Bed(element = Q.ufl_element())

class Depth(Expression):
  def eval(self, values, x):
    values[0] = min(0, x[2])
D = Depth(element = Q.ufl_element())

class Beta(Expression):
  def eval(self, values, x):
    values[0] = bm * gauss(x[0], x[1], L/2, L/2)
beta = Beta(element = Q.ufl_element())

xmin = -L
xmax = L
ymin = -L
ymax = L

# width and origin of the domain for deforming x coord :
width_x  = xmax - xmin
offset_x = xmin

# width and origin of the domain for deforming y coord :
width_y  = ymax - ymin
offset_y = ymin

# Deform the mesh to the defined geometry :
for x in mesh.coordinates():
  # transform x :
  x[0]  = x[0]  * width_x

  # transform y :
  x[1]  = x[1]  * width_y

  # transform z :
  # thickness = surface - base, z = thickness + base
  x[2]  = x[2] * (S(x[0], x[1], x[2]) - B(x[0], x[1], x[2]))
  x[2]  = x[2] +  B(x[0], x[1], x[2])

# constants :
p0     = 101325
cp     = 1007
T0     = 288.15
M      = 0.0289644
rho    = 917.0
rho_w  = 1000.0
g      = 9.80665
gamma  = 8.71e-4
E      = 1.0
R      = 8.31447
n      = 3.0
T      = 250.0
x      = SpatialCoordinate(mesh)
N      = FacetNormal(mesh)
h      = CellSize(mesh)
I      = Identity(3)

# solver parameters :
parameters['form_compiler']['quadrature_degree'] = 3
params = {"newton_solver":
         {"linear_solver"        : 'mumps',
          "preconditioner"       : 'default',
          "maximum_iterations"   : 35,
          "relaxation_parameter" : 1.0,
          "relative_tolerance"   : 1e-3,
          "absolute_tolerance"   : 1e-16}}

# create functions for boundary conditions :
inflow = Expression("200*sin(x[1]*pi/L)", L=L)
f_w    = rho*g*(S - x[2]) + rho_w*g*D
p_a    = p0 * (1 - g*x[2]/(cp*T0))**(cp*M/R)

# boundary condition for velocity :
bc1 = DirichletBC(W.sub(1),        0.0, ff, 1)  # pressure
bc2 = DirichletBC(W.sub(0).sub(2), 0.0, ff, 1)  # w on surface

bcs = [bc1, bc2]

#===============================================================================
# define variational problem :
F   = Function(W)
dU  = TrialFunction(W)
Tst = TestFunction(W)

U,   P  = split(F)
Phi, xi = split(Tst)

u,   v,   w   = U
phi, psi, chi = Phi

# rate-factor :
Tstar = T + gamma * (S - x[2])
a_T   = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10)
Q_T   = conditional( lt(Tstar, 263.15), 6e4,          13.9e4)
A     = E * a_T * exp( -Q_T / (R * Tstar))
b     = A**(-1/n)

# gravity vector :
gv = as_vector([0, 0, g])

# Second invariant of the strain rate tensor squared
epi   = strain_rate(U)
ep_xx = epi[0,0]
ep_yy = epi[1,1]
ep_xy = epi[0,1]
ep_xz = epi[0,2]
ep_yz = epi[1,2]

epsdot = ep_xx**2 + ep_yy**2 + ep_xx*ep_yy + ep_xy**2 + ep_xz**2 + ep_yz**2
eta    = 0.5 * b * (epsdot + 1e-10)**((1-n)/(2*n))
eta    = 1e8

sigma   = 2*eta*epi - P*I
beta   = Constant(1e5)

# conservation of momentum :
R1 = + inner(sigma, grad(Phi)) * dx \
     + rho * dot(gv, Phi) * dx \
     + beta**2 * dot(U, Phi) * dBed \
     - f_w * dot(N, Phi) * dGamma \
     - p_a * dot(N, Phi) * dSrf \

# conservation of mass :
R2 = + div(U) * xi * dx \

# total residual :
R = R1 + R2

# Jacobian :
J = derivative(R, F, dU)

# compute solution :
solve(R == 0, F, bcs, J=J, solver_parameters=params)

File("output/U.pvd")    << project(U)
File("output/P.pvd")    << project(P)
File("output/divU.pvd") << project(div(U))
File("output/beta.pvd") << interpolate(beta,Q)



