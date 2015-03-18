from dolfin import *
from pylab  import *

mesh = UnitCubeMesh(10,10,10)          # original mesh
mesh.coordinates()[:,0] -= .5          # shift x-coords
mesh.coordinates()[:,1] -= .5          # shift y-coords
V    = FunctionSpace(mesh, "CG", 1)
u    = Function(V)

# apply expression over cube for clearer results :
u_i  = Expression('sqrt(pow(x[0],2) + pow(x[1], 2))')
u.interpolate(u_i)

bmesh  = BoundaryMesh(mesh, "exterior")   # surface boundary mesh

# mark the boundary of the bottom surface :
cellmap = bmesh.entity_map(2)
vertmap = bmesh.entity_map(0)
pb      = CellFunction("size_t", bmesh, 0)
for c in cells(bmesh):
  if Facet(mesh, cellmap[c.index()]).normal().z() < 0:
    pb[c] = 1

submesh = SubMesh(bmesh, pb, 1)           # bottom of boundary mesh

Vb  = FunctionSpace(bmesh,   "CG", 1)     # surface function space
Vs  = FunctionSpace(submesh, "CG", 1)     # submesh function space

ub  = Function(Vb)                        # boundary function
us  = Function(Vs)                        # surface function

ub.interpolate(u)                         # interpolate u onto boundary
us.interpolate(u)                         # interpolate u onto surface mesh

unb = Function(Vb)                        # new boundary function
un  = Function(V)                         # new whole function


# mappings we may need :
m    = vertex_to_dof_map(V)
b    = vertex_to_dof_map(Vb)
s    = vertex_to_dof_map(Vs)
                            
mi   = dof_to_vertex_map(V)
bi   = dof_to_vertex_map(Vb)
si   = dof_to_vertex_map(Vs)

# mapping from submesh back to bmesh :
t = submesh.data().array('parent_vertex_indices', 0)

# get vertex-valued arrays :
us_a  = us.vector().array()
u_a   = u.vector().array()
ub_a  = ub.vector().array()

unb_a = unb.vector().array()
un_a  = un.vector().array()

# update the values of the new functions to be the values of the surface :
unb_a[b[t]]  = us_a[s]   # works

un_a[m[b[t]]] = us_a[s]  # need something to make this sort of thing work

un.vector().set_local(un_a)
unb.vector().set_local(unb_a)

# save for viewing :
File("output/u.pvd")      << u
File("output/ub_n.pvd")   << unb
File("output/un.pvd")     << un
File("output/us.pvd")     << us



