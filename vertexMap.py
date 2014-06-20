from fenics import *

n      = 3
mesh   = BoxMesh(-1,-1,-1,1,1,1,n,n,n)
f_mesh = Mesh(mesh)

Q    = FunctionSpace(  mesh, "CG", 1)
Q_f  = FunctionSpace(f_mesh, "CG", 1)
u    = Function(Q)
f    = Expression('pow(x[0],2) + pow(x[1],2) + pow(x[2],2)')
f    = interpolate(f,Q_f)
f_a  = f.vector().array()

v2d  = vertex_to_dof_map(Q_f)
d2v  = dof_to_vertex_map(Q)

u.vector().set_local(f_a)
u.vector().apply('insert')

u_v  = u.compute_vertex_values()

u.vector().set_local(u_v[d2v])
u.vector().apply('insert')

File('output/u.pvd') << u
