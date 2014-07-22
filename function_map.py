from dolfin import *

mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'CG', 1)

cf = CellFunction('size_t', mesh, 0)
o1 = AutoSubDomain(lambda x : x[1] >= x[0] - DOLFIN_EPS)
o1.mark(cf, 1)

mu = 100

# using dofmap
dofmap = V.dofmap()
o0_dofs = []
o1_dofs = []

for cell in cells(mesh): # compute dofs in the domains
    if cf[cell] == 0:
        o0_dofs.extend(dofmap.cell_dofs(cell.index()))
    else:
        o1_dofs.extend(dofmap.cell_dofs(cell.index()))

# unique
o0_dofs = list(set(o0_dofs))
o1_dofs = list(set(o1_dofs))

u = interpolate(Expression("x[0]*x[0]"), V)
u1 = Function(V)

u1.vector()[o0_dofs] = u.vector()[o0_dofs]
u1.vector()[o1_dofs] = mu*u.vector()[o1_dofs]

#using characteristic function
v1 = Function(V)

chi0 = Function(V)
chi1 = Function(V)

for cell in cells(mesh): # set the characteristic functions
    if cf[cell] == 0:
        chi0.vector()[dofmap.cell_dofs(cell.index())] = 1
    else:
        chi1.vector()[dofmap.cell_dofs(cell.index())] = 1

v1 = project(chi0*u, V)
v1 += project(chi1*mu*u, V)


File('output/u1.pvd') << u1
File('output/v1.pvd') << project(v1, V)




