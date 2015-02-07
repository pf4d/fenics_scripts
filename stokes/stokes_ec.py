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
# Modified by Anders Logg, 2008-2009.
# Modified by Evan Cummings, 2015
#
# First added:  2007-11-16
# Last changed: 2015-02-06
# Begin demo

from dolfin import *

l    = 10000
mesh = RectangleMesh(0, 0, l, 1, 100, 10)
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

ds       = Measure('ds')[ff]
dSrf     = ds(1)
dBed     = ds(2)
dLeft    = ds(3)
dRight   = ds(4)

# Deform the square to the defined geometry :
for x in mesh.coordinates():
  # transform z :
  # thickness = surface - base, z = thickness + base
  x[1]  = x[1] * S(x[0], x[1])

rho = 917.9
g   = 9.8
eta = 1.0

# boundary conditions : 
noslip = Constant((0, 0))
bc0    = DirichletBC(W.sub(0), noslip, ff, 2)  # zero velocity on bed
bc1    = DirichletBC(W.sub(0), noslip, ff, 4)  # zero velocity at divide
bc2    = DirichletBC(W.sub(1), 0.0,    ff, 1)  # zero pressure on surface

# Collect boundary conditions
bcs = [bc0, bc1]

# Define variational problem
I = Identity(2)
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

def epsilon(u): return 0.5*(grad(u) + grad(u).T)
def sigma(u,p): return 2*eta*epsilon(u) - p*I
def L(u,p):     return -div(sigma(u,p))

f = Constant((0, -rho * g))
a = inner(sigma(u,p), grad(v))*dx + q*div(u)*dx
L = inner(f,v)*dx + rho*g*(S - x[1])*dot(v,n)*dRight

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

print "Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2")

# # Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
File("output/velocity.pvd") << u
File("output/pressure.pvd") << p



