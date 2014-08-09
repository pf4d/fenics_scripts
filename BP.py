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
  L      = u * phi * dx                  # lhs
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

nx        = 80
ny        = 80
nz        = 5
mesh      = UnitCubeMesh(nx,ny,nz)
flat_mesh = Mesh(mesh)

# Define function spaces
Q = FunctionSpace(mesh, "CG", 1)
V = MixedFunctionSpace([Q,Q])

ff   = FacetFunction('size_t', mesh, 0)

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
dBed   = ds(2)
dGamma = ds(3) + ds(4) + ds(5) + ds(6)

alpha   = 0.5 * pi / 180
L       = 5000.0
S0      = 200.0

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
    b         = B(x[0], x[1], x[2])
    values[0] = abs(min(0, b))
D = Depth(element = Q.ufl_element())

class Beta(Expression):
  def eval(self, values, x):
    values[0] = 500 + 500 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)
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
for x,x0 in zip(mesh.coordinates(), flat_mesh.coordinates()):
  # transform x :
  x[0]  = x[0]  * width_x + offset_x
  x0[0] = x0[0] * width_x + offset_x

  # transform y :
  x[1]  = x[1]  * width_y + offset_y
  x0[1] = x0[1] * width_y + offset_y

  # transform z :
  # thickness = surface - base, z = thickness + base
  x[2]  = x[2] * (S(x[0], x[1], x[2]) - \
                  B(x[0], x[1], x[2]))
  x[2]  = x[2] + B(x[0], x[1], x[2])

# constants :
eta    = 1e8
rho    = 917.0
rho_w  = 1000.0
g      = 9.81
x      = SpatialCoordinate(mesh)
N      = FacetNormal(mesh)

# solver parameters :
params = {"newton_solver":
         {"relative_tolerance": 1e-6,
          "absolute_tolerance": 1e-21}}

# create functions for boundary conditions :
noslip = Constant((0, 0))
inflow = Expression(("50*sin(x[1]*pi/L)", "0"), L=L)
zero   = Constant(0)
f_w    = rho*g*(S + x[2]) + rho_w*g*D

# boundary condition for velocity :
bc1 = DirichletBC(V, noslip, ff, 5)
bc2 = DirichletBC(V, noslip, ff, 6)
bc3 = DirichletBC(V, inflow, ff, 3)
bcs = [bc1, bc2, bc3]


#===============================================================================
# define variational problem :
U   = Function(V)
u,v = split(U)

dU       = TrialFunction(V)
Phi      = TestFunction(V)
phi, psi = split(Phi)

epi_1  = as_vector([2*u.dx(0) + v.dx(1), 
                    0.5*(u.dx(1) + v.dx(0)),
                    0.5*u.dx(2)             ])
epi_2  = as_vector([0.5*(u.dx(1) + v.dx(0)),
                         u.dx(0) + 2*v.dx(1),
                    0.5*v.dx(2)             ])
gradS  = as_vector([S.dx(0), S.dx(1)])
n      = as_vector([N[0],    N[1]])

F = + 2 * eta * dot(epi_1, grad(phi)) * dx \
    + 2 * eta * dot(epi_2, grad(psi)) * dx \
    + rho * g * dot(gradS, Phi) * dx \
    + beta**2 * dot(U, Phi) * dBed \
    - f_w * dot(n, Phi) * dGamma

J = derivative(F, U, dU)

# compute solution :
solve(F == 0, U, bcs, J=J, solver_parameters=params)


#===============================================================================
# define variational problem :
Uhat      = Function(V)
uhat,vhat = split(U)

dU       = TrialFunction(V)
Phi      = TestFunction(V)
phi, psi = split(Phi)

U_te    = as_vector([project(u), project(v)])
U_nm    = normalize_vector(U_te)
U_n     = as_vector([ U_nm[0], U_nm[1]])
U_t     = as_vector([-U_nm[1], U_nm[0]])

u_s     = dot(Uhat, U_n)
v_s     = dot(Uhat, U_t)
U_s     = as_vector([u_s,       v_s])
gradu   = as_vector([u_s.dx(0), u_s.dx(1)])
gradv   = as_vector([v_s.dx(0), v_s.dx(1)])
dudn    = dot(gradu, U_n)
dudt    = dot(gradu, U_t)
dudz    = u_s.dx(2)
dvdn    = dot(gradv, U_n)
dvdt    = dot(gradv, U_t)
dvdz    = v_s.dx(2)
gradphi = as_vector([phi.dx(0), phi.dx(1)])
gradpsi = as_vector([psi.dx(0), psi.dx(1)])
gradS   = as_vector([S.dx(0),   S.dx(1)])
dphidn  = dot(gradphi, U_n)
dphidt  = dot(gradphi, U_t)
dpsidn  = dot(gradpsi, U_n)
dpsidt  = dot(gradpsi, U_t)
gradphi = as_vector([dphidn,  dphidt, phi.dx(2)])
gradpsi = as_vector([dpsidn,  dpsidt, psi.dx(2)])
dSdn    = dot(gradS, U_n)
dSdt    = dot(gradS, U_t)
gradS   = as_vector([dSdn,  dSdt])
n       = as_vector([N[0],  N[1]])

epi_1  = as_vector([2*dudn + dvdt, 
                    0.5*(dudt + dvdn),
                    0.5*dudz             ])
epi_2  = as_vector([0.5*(dudt + dvdn),
                    dudn + 2*dvdt,
                    0.5*dvdz             ])

F = + 2 * eta * dot(epi_1, gradphi) * dx \
    + 2 * eta * dot(epi_2, gradpsi) * dx \
    + rho * g * dot(gradS, Phi) * dx \
    + beta**2 * dot(U_s, Phi) * dBed \
    + f_w * dot(n, Phi) * dGamma

J = derivative(F, Uhat, dU)

# compute solution :
#solve(F == 0, Uhat, bcs, J=J, solver_parameters=params)


#===============================================================================
# define variational problem :
phi     = TestFunction(Q)
dtau    = TrialFunction(Q)
        
#U_te    = as_vector([project(uhat), project(vhat)])
#U_nm    = normalize_vector(U_te)
#U_n     = as_vector([U_nm[0],  U_nm[1]])
#U_t     = as_vector([U_nm[1], -U_nm[0]])

u_s     = dot(U, U_n)
v_s     = dot(U, U_t)
gradu   = as_vector([u_s.dx(0), u_s.dx(1)])
gradv   = as_vector([v_s.dx(0), v_s.dx(1)])
dudn    = dot(gradu, U_n)
dudt    = dot(gradu, U_t)
dudz    = u_s.dx(2)
dvdn    = dot(gradv, U_n)
dvdt    = dot(gradv, U_t)
dvdz    = v_s.dx(2)
gradphi = as_vector([phi.dx(0), phi.dx(1)])
gradS   = as_vector([S.dx(0),   S.dx(1)])
dphidn  = dot(gradphi, U_n)
dphidt  = dot(gradphi, U_t)
gradphi = as_vector([dphidn,  dphidt, phi.dx(2)])
dSdn    = dot(gradS, U_n)
dSdt    = dot(gradS, U_t)
gradS   = as_vector([dSdn,  dSdt])
n       = as_vector([N[0],  N[1]])

epi_1  = as_vector([2*dudn + dvdt, 
                    0.5*(dudt + dvdn),
                    0.5*dudz             ])
epi_2  = as_vector([0.5*(dudt + dvdn),
                         dudn + 2*dvdt,
                    0.5*dvdz             ])

# driving stres :
tau_dn = phi * rho * g * (S - B) * dSdn * dx
tau_dt = phi * rho * g * (S - B) * dSdt * dx

# calc basal drag : 
tau_bn = beta**2 * u_s * phi * dBed
tau_bt = beta**2 * v_s * phi * dBed

# stokes equation weak form in normal dir. (n) and tangent dir. (t) :
tau_nn = - dphidn * eta * (4*dudn + 2*dvdt) * dx
tau_nt = - dphidt * eta * (  dudt +   dvdn) * dx
tau_tn = - dphidn * eta * (  dudt +   dvdn) * dx
tau_tt = - dphidt * eta * (4*dvdt + 2*dudn) * dx

# mass matrix :
M = assemble(phi*dtau*dx)

# assemble the vectors :
tau_dn_v   = assemble(tau_dn)
tau_dt_v   = assemble(tau_dt)
tau_bn_v   = assemble(tau_bn)
tau_bt_v   = assemble(tau_bt)
tau_nn_v   = assemble(tau_nn)
tau_nt_v   = assemble(tau_nt)
tau_tn_v   = assemble(tau_tn)
tau_tt_v   = assemble(tau_tt)

# solution functions :
tau_dn   = Function(Q)
tau_dt   = Function(Q)
tau_bn   = Function(Q)
tau_bt   = Function(Q)
tau_nn   = Function(Q)
tau_nt   = Function(Q)
tau_tn   = Function(Q)
tau_tt   = Function(Q)

# solve the linear system :
solve(M, tau_dn.vector(),   tau_dn_v)
solve(M, tau_dt.vector(),   tau_dt_v)
solve(M, tau_bn.vector(),   tau_bn_v)
solve(M, tau_bt.vector(),   tau_bt_v)
solve(M, tau_nn.vector(),   tau_nn_v)
solve(M, tau_nt.vector(),   tau_nt_v)
solve(M, tau_tn.vector(),   tau_tn_v)
solve(M, tau_tt.vector(),   tau_tt_v)

# calculate the residual :
tau_nn   = vert_integrate(tau_nn, Q, ff)
tau_nn   = extrude(tau_nn, 1, 2, Q, ff)
tau_nt   = vert_integrate(tau_nt, Q, ff)
tau_nt   = extrude(tau_nt, 1, 2, Q, ff)
tau_bn   = extrude(tau_bn, 2, 2, Q, ff)
tau_totn = project(tau_nn + tau_nt - tau_bn - tau_dn)

#===============================================================================
# save solution in VTK format :
#File('output/ff.pvd')       << ff
#File("output/S.pvd")        << interpolate(S,Q)
#File("output/B.pvd")        << interpolate(B,Q)
#File("output/U_nm.pvd")     << project(U_nm)
File("output/U.pvd")        << U
File("output/Uhat.pvd")     << Uhat
File("output/tau_dn.pvd")   << tau_dn
#File("output/tau_dt.pvd")   << tau_dt
File("output/tau_bn.pvd")   << tau_bn
#File("output/tau_bt.pvd")   << tau_bt
File("output/tau_nn.pvd")   << tau_nn
File("output/tau_nt.pvd")   << tau_nt
File("output/tau_totn.pvd") << tau_totn
#File("output/tau_tn.pvd")   << tau_tn
#File("output/tau_tt.pvd")   << tau_tt



