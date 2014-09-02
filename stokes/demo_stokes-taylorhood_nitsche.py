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

# define measures :
ds   = Measure('ds')[sub_domains]
dG_0 = ds(0)
dG_r = ds(1)
dG_l = ds(2)
dGamma = ds(0) + ds(1)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
N = FacetNormal(mesh)
I = Identity(2)
W = V * Q

# No-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
u0_0 = Expression(("0", "0"), element=V.ufl_element())

# Inflow boundary condition for velocity
# x0 = 1
u0_r = Expression(("-sin(x[1]*pi)", "0.0"), element=V.ufl_element())

# Boundary condition for pressure at outflow
# x0 = 0
p0_l = Constant(0)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

f = Constant((0, 0))

C_b   = 500
beta  = 2 * C_b**2
C_a   = 500
alpha = 2 * C_a**2 
C_d   = 500
delta = 2 * C_d**2
 

a = + inner(grad(u) - p*I, grad(v)) * dx \
    + q*div(u) * dx \
    - inner(dot(grad(u), N), v) * (dG_r + dG_0) \
    - inner(dot(grad(v), N), u) * (dG_r + dG_0) \
    + beta**2*dot(v,N)*dot(u,N) * dG_0 \
    + beta*inner(v,u) * dG_0 \
    + delta*inner(v,u) * dG_r \
    + alpha*p*div(v) * dG_l \

L = + inner(f, v)*dx \
    - inner(dot(grad(v), N), u0_r) * dG_r \
    - inner(dot(grad(v), N), u0_0) * dG_0 \
    + beta*inner(u0_0,v) * dG_0 \
    + delta*inner(u0_r,v) * dG_r \
    + alpha*p0_l*div(v) * dG_l \

# Compute solution
w = Function(W)
solve(a == L, w)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

print "Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2")

# # Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
File("output/v_nit.pvd") << u
File("output/p_nit.pvd") << p



