# This demo solves the Stokes equations, using stabilized
# first order elements for the velocity and pressure. The
# sub domains for the different boundary conditions used
# in this simulation are computed by the demo program in
# src/demo/mesh/subdomains.
#
# Original implementation: ../cpp/main.cpp by Anders Logg

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
# Modified by Evan Cummings, 2014.
#
# First added:  2007-11-15
# Last changed: 2014-06-11
# Begin demo

from dolfin import *

mesh      = UnitCubeMesh(10,10,10)
flat_mesh = Mesh(mesh)

# Define function spaces
scalar = FunctionSpace(mesh, "CG", 1)
vector = VectorFunctionSpace(mesh, "CG", 1)
system = vector * scalar

ff   = FacetFunction('size_t', mesh, 0)

# iterate through the facets and mark each if on a boundary :
#
#   2 = high slope, upward facing ................ surface
#   3 = high slope, downward facing .............. base
#   4 = low slope, upward or downward facing ..... right side
#   5 = low slope, upward or downward facing ..... left side
for f in facets(mesh):
  n       = f.normal()    # unit normal vector to facet f
  tol     = 1e-3

  if   n.z() >=  tol and f.exterior():
    ff[f] = 2

  elif n.z() <= -tol and f.exterior():
    ff[f] = 3

  elif n.z() >  -tol and n.z() < tol and f.exterior() \
                     and n.x() > tol and n.y() < tol :
    ff[f] = 4

  elif n.z() >  -tol and n.z() < tol  and f.exterior() \
                     and n.x() < -tol and n.y() < tol :
    ff[f] = 5

ds = Measure('ds')[ff]

File('ff.pvd') << ff


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

S = Surface(element = scalar.ufl_element())
B = Bed(element = scalar.ufl_element())
H = S - B

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

# Create functions for boundary conditions
noslip = Constant((0, 0, 0))
inflow = Expression(("-sin(2*x[1]*pi/L)", "0", "0"), L=L)
zero   = Constant(0)

# No-slip boundary condition for velocity
bc0 = DirichletBC(system.sub(0), noslip, ff, 3)

# Inflow boundary condition for velocity
bc1 = DirichletBC(system.sub(0), inflow, ff, 4)

# Boundary condition for pressure at outflow
bc2 = DirichletBC(system.sub(1), zero, ff, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

## Define variational problem
#w   = Function(system)
#U,p = split(w)
#
#Phi      = TestFunction(system)
#phi, psi = split(Phi)
#
#dw  = TrialFunction(system)
#
#f = Constant((0, 0, 0))
#h = CellSize(mesh)
#beta  = 0.2
#delta = beta*h*h
#a = (inner(grad(phi), grad(U)) - div(phi)*p + psi*div(U) + \
#    delta*inner(grad(psi), grad(p)))*dx
#L = inner(phi + delta*grad(psi), f)*dx
#
#F = a - L
#
#J = derivative(F, w, dw)
#
## Compute solution
#solve(F == 0, w, bcs, J=J, solver_parameters={"newton_solver":
#                                             {"relative_tolerance": 1e-16,
#                                              "absolute_tolerance": 1e-21}})
#u, p = w.split()
#
#(v, q) = TestFunctions(system)
#(u, p) = TrialFunctions(system)
#f      = Constant((0, 0, 0))
#h      = CellSize(mesh)
#beta   = 0.2
#delta  = beta*h*h
#a      = (inner(grad(v), grad(u)) - div(v)*p + q*div(u) + \
#         delta*inner(grad(q), grad(p)))*dx
#L      = inner(v + delta*grad(q), f)*dx

(v, q) = TestFunctions(system)
(u, p) = TrialFunctions(system)

#eta    = 1e6
#rho    = 917.0
#g      = 9.81
#epsdot = 0.5 * (grad(u) + grad(u).T)
#tau    = 2 * eta * epsdot
#
#tau_xx = tau[0,0]
#tau_yy = tau[1,1]
#tau_xy = tau[0,1]
#tau_yx = tau_xy
#tau_xz = tau[0,2]
#tau_yz = tau[1,2]
#tau_zz = tau[2,2]
#
#epi_1  = as_vector([2*tau_xx + tau_yy, 
#                    tau_xy,
#                    tau_xz])
#epi_2  = as_vector([2*tau_yy + tau_xx,
#                    tau_yx,
#                    tau_yz])
#
#f     = rho * g * grad(S)

eta    = 1e6
rho    = 917.0
g      = as_vector([0, 0, -9.81])
epsdot = (grad(u) + grad(u).T)
tau    = eta * epsdot


h     = CellSize(mesh)
beta  = 0.2
delta = beta*h*h

f     = rho * g

a     = (inner(grad(v), tau) - div(v)*p + q*div(u) + \
        delta*inner(grad(q), grad(p)))*dx
L     = inner(v + delta*grad(q), f)*dx

# Compute solution
w = Function(system)
solve(a == L, w, bcs)
u, p = w.split()

# Save solution in VTK format
File("output/velocity.pvd") << u
File("output/pressure.pvd") << p

