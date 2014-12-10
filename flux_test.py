from pylab    import *
from fenics   import *
from matrices import plot_matrix

n    = 2
#mesh = UnitIntervalMesh(3)
mesh = UnitSquareMesh(n,n)
#mesh = UnitCubeMesh(n,n,n)
#mesh = Mesh('meshes/unit_square_mesh.xml')
#mesh = Mesh('meshes/unit_cube_mesh.xml')

# refine mesh :
origin = Point(0.0,0.5,0.0)
for i in range(1,1):
  cell_markers = CellFunction("bool", mesh)
  cell_markers.set_all(False)
  for cell in cells(mesh):
    p = cell.midpoint()
    if p.distance(origin) < 1.0/i:
      cell_markers[cell] = True
  mesh = refine(mesh, cell_markers)

Q = FunctionSpace(mesh, 'CG', 1)
V = VectorFunctionSpace(mesh, 'CG', 1)

w = TrialFunction(Q)
v = TestFunction(Q)
u = Function(Q)

f = Constant(1.0)

a = w.dx(0) * v.dx(0) * dx
l = f * v * dx

def left(x, on_boundary):
  return x[0] == 0 and on_boundary

bc = DirichletBC(Q, 0.0, left)

solve(a == l, u, bc)

File('output/u.pvd') << u

uv = u.vector().array()
b  = assemble(l).array()
A  = assemble(a).array()
h  = project(CellSize(mesh),Q).vector().array()

t  = np.dot(A,uv) - b

q = Function(Q)
q.vector().set_local(t)
q.vector().apply('insert')

File('output/q.pvd') << q

fig = figure()
ax  = fig.add_subplot(111)

plot_matrix(A, ax, r'$K$ - stiffness matrix', continuous=False)

tight_layout()
show()



