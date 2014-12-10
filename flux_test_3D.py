from pylab    import *
from fenics   import *
from matrices import plot_matrix

n    = 10
mesh = UnitCubeMesh(n,n,n)
#mesh = Mesh('meshes/unit_cube_mesh.xml')

# refine mesh :
for i in range(1,1):
  cell_markers = CellFunction("bool", mesh)
  cell_markers.set_all(False)
  for cell in cells(mesh):
    p = cell.midpoint()
    if p.y() <= 1.0/i:
      cell_markers[cell] = True
  mesh = refine(mesh, cell_markers)

Q = FunctionSpace(mesh, 'CG', 1)

w = TrialFunction(Q)
v = TestFunction(Q)
u = Function(Q, name='u')

f = Constant(1.0)

a = inner(grad(w), grad(v)) * dx
l = f * v * dx

def left(x, on_boundary):
  return x[0] == 0 and on_boundary

def right(x, on_boundary):
  return x[0] == 1 and on_boundary

def top(x, on_boundary):
  return x[1] == 1 and on_boundary

def bottom(x, on_boundary):
  return x[1] == 0 and on_boundary

bcl = DirichletBC(Q, 0.0, left)
bcr = DirichletBC(Q, 0.0, right)
bct = DirichletBC(Q, 0.0, top)
bcb = DirichletBC(Q, 0.0, bottom)

solve(a == l, u, bcl)

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



