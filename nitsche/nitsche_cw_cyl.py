#! /usr/bin/env python
"""
Demo for Nitsche-type free-slip boundary conditions
"""

__author__ = "Christian Waluga (waluga@ma.tum.de)"
__copyright__ = "Copyright (c) 2013 %s" % __author__

from dolfin import *

# avoid outputs like 'solving variational problem' by FEniCS
set_log_level(ERROR)

mesh = refine(Mesh("cylinderbump.xml"))

# naive equal order element
V = VectorFunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)

mu = Constant(1.0)
F = Constant((0.0,0.0))

W = V*Q

# variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# boundary conditions
leftexp = Expression(("x[1]*(3-x[1])/10", "0.0"))
left = DirichletBC(W.sub(0), leftexp, "near(x[0], -5)")
top = DirichletBC(W.sub(0), (0,0), "near(x[1], 3)")
bcs = [left, top]

class Bottom(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and not (near(x[0], -5) or near(x[0], 5) or near(x[1], 3))
bottom = Bottom()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
bottom.mark(boundaries, 1)
ds = Measure("ds")[boundaries]

alpha = Constant(1./10)
beta = Constant(10)
h = CellSize(mesh)
n = FacetNormal(mesh)

# (bi)linear forms
def a(u,v): return inner(mu*grad(u),grad(v))*dx
def b(v,q): return - div(v)*q*dx
def f(v):   return dot(F, v)*dx
def t(u,p): return dot(2*mu*sym(grad(u)),n) - p*n

stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       - dot(n,t(u,p))*dot(v,n)*ds(1) - dot(u,n)*dot(n,t(v,q))*ds(1) \
       + beta/h*dot(u,n)*dot(v,n)*ds(1) \
       + alpha*h**2*dot(F - grad(p), grad(q))*dx

# solve variational problem
wh = Function(W)
print 'size:', wh.vector().size()

solve(lhs(stokes) == rhs(stokes), wh, bcs, \
  solver_parameters = { \
    "linear_solver": "tfqmr", \
    "preconditioner": "default"} )

uh, ph = wh.split()
plot(uh, interactive = True)
File("cylinderbump-velocity.pvd") << uh
File("cylinderbump-pressure.pvd") << ph

exit()
