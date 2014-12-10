from pylab    import *
from fenics   import *
from matrices import plot_matrix

n    = 200
mesh = UnitIntervalMesh(3)

# refine mesh :
origin = Point(0.0,0.0,0.0)
for i in range(1,1):
  cell_markers = CellFunction("bool", mesh)
  cell_markers.set_all(False)
  for cell in cells(mesh):
    p = cell.midpoint()
    if p.distance(origin) < 1.0/i:
      cell_markers[cell] = True
  mesh = refine(mesh, cell_markers)

Q = FunctionSpace(mesh, 'CG', 1)

w = TrialFunction(Q)
v = TestFunction(Q)
u = Function(Q, name='u')

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

t  = np.dot(A,uv) - b

q = Function(Q, name='q')
q.vector().set_local(t)
q.vector().apply('insert')

File('output/q.pvd') << q

fig = figure()
ax  = fig.add_subplot(111)

plot_matrix(A, ax, r'stiffness matrix $K$', continuous=False)

tight_layout()
show()



