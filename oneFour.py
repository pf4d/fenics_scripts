from fenics import *

mesh      = UnitCubeMesh(10,10,10)
flat_mesh = Mesh(mesh)

Q    = FunctionSpace(mesh, 'CG', 1)
Q2   = MixedFunctionSpace([Q,Q])
ff   = FacetFunction('size_t', mesh, 0)

# iterate through the facets and mark each if on a boundary :
#
#   2 = high slope, upward facing ................ surface
#   3 = high slope, downward facing .............. base
#   4 = low slope, upward or downward facing ..... sides
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = 1e-3

  if   n.z() >=  tol and f.exterior():
    ff[f] = 2

  elif n.z() <= -tol and f.exterior():
    ff[f] = 3

  elif n.z() >  -tol and n.z() < tol and f.exterior():
    ff[f] = 4

ds = Measure('ds')[ff]


alpha   = 0.5 * pi / 180
L       = 5000.0

class Surface(Expression):
  def eval(self,values,x):
    values[0] = - x[0] * tan(alpha)


class Bed(Expression):
  def eval(self,values,x):
    values[0] = - x[0] * tan(alpha) \
                - 1000.0 \
                + 500.0 * sin(2*pi*x[0]/L) * sin(2*pi*x[1]/L)

S = Surface(element = Q.ufl_element())
B = Bed(element = Q.ufl_element())

xmin = 0
xmax = L
ymin = 0
ymax = L

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

# set the parameters
stokes_params = NonlinearVariationalSolver.default_parameters()
#stokes_params['newton_solver']['method']                  = 'lu'
#stokes_params['newton_solver']['report']                  = True
#stokes_params['newton_solver']['relaxation_parameter']    = 1.0
#stokes_params['newton_solver']['relative_tolerance']      = 2e-12
#stokes_params['newton_solver']['absolute_tolerance']      = 1e-3
#stokes_params['newton_solver']['maximum_iterations']      = 25
#stokes_params['newton_solver']['error_on_nonconvergence'] = False
#stokes_params['newton_solver']['linear_solver']           = 'mumps'
#stokes_params['newton_solver']['preconditioner']          = 'default'
parameters['form_compiler']['quadrature_degree']          = 2

r             = 1.0
n             = 3.0
gamma         = 8.71e-4
R             = 8.314
eps_reg       = 1e-15
rho           = 910.0
rho_w         = 1000.0
g             = 9.81
T0            = 268.0

S             = interpolate(S, Q)
B             = interpolate(B, Q)
x             = SpatialCoordinate(mesh)

# initialize the temperature :
T     = Function(Q)
T.vector()[:] = 268.0
  
# initialize the bed friction coefficient :
beta2 = Function(Q)
beta2.vector()[:] = 0.5

# initialize the enhancement factor :
E = Function(Q)
E.vector()[:] = 1.0

# Define a test function
Phi      = TestFunction(Q2)

# Define a trial function
dU       = TrialFunction(Q2)
U        = Function(Q2)

phi, psi = split(Phi)
du,  dv  = split(dU)
u,   v   = split(U)     # x,y velocity components
w        = Function(Q)  # z   velocity component

# vertical velocity components :
chi      = TestFunction(Q)
dw       = TrialFunction(Q)

dSurf    = ds(2)      # surface
dGrnd    = ds(3)      # bed

# Define pressure corrected temperature
Tstar = T + gamma * (S - x[2])
 
# Define ice hardness parameterization :
a_T   = conditional( lt(Tstar, 263.15), 1.1384496e-5, 5.45e10)
Q_T   = conditional( lt(Tstar, 263.15), 6e4,13.9e4)
b     = ( E * a_T * exp( -Q_T / (R * Tstar)) )**(-1/n)

# second invariant of the strain rate tensor squared :
term     = + 0.5 * (u.dx(2)**2 + v.dx(2)**2 + (u.dx(1) + v.dx(0))**2) \
           +        u.dx(0)**2 + v.dx(1)**2 + (u.dx(0) + v.dx(1))**2
epsdot   =   0.5 * term + eps_reg
eta      =     b * epsdot**((1.0 - n) / (2*n))

# 1) Viscous dissipation
Vd       = (2*n)/(n+1) * b * epsdot**((n+1)/(2*n))

# 2) Potential energy
Pe       = rho * g * (u * S.dx(0) + v * S.dx(1))

# 3) Dissipation by sliding
Sl       = 0.5 * beta2 * (S - B)**r * (u**2 + v**2)

# Variational principle
A        = (Vd + Pe)*dx + Sl*dGrnd

# Calculate the first variation (the action) of the variational 
# principle in the direction of the test function
F   = derivative(A, U, Phi)

# Calculate the first variation of the action (the Jacobian) in
# the direction of a small perturbation in U
J   = derivative(F, U, dU)

# vertical velocity residual :
w_R = (u.dx(0) + v.dx(1) + dw.dx(2))*chi*dx - \
      (u*B.dx(0) + v*B.dx(1) - dw)*chi*dGrnd

# solve nonlinear system :
print "::: solving velocity :::"
solve(F == 0, U, J = J, solver_parameters = stokes_params)

# solve for vertical velocity :
solve(lhs(w_R) == rhs(w_R), w)

File('U.pvd') << project(as_vector([u,v,w]))



