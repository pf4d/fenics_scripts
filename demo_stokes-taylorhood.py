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
#
# First added:  2007-11-16
# Last changed: 2009-11-26
# Begin demo

from dolfin import *

# Load mesh and subdomains
mesh = Mesh("meshes/dolfin_fine.xml.gz")
sub_domains = MeshFunction("size_t", mesh, 
                           "meshes/dolfin_fine_subdomains.xml.gz")

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
I = Identity(2)
W = V * Q

# No-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
# x0 = 0
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u) - p*I, grad(v)) + q*div(u))*dx
L = inner(f, v)*dx

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



