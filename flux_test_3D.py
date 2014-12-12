from pylab    import *
from fenics   import *
from matrices import plot_matrix

def get_facet_normal(bmesh):
  '''Manually calculate FacetNormal function'''

  if not bmesh.type().dim() == 2:
    raise ValueError('Only works for 2-D mesh')

  vertices = bmesh.coordinates()
  cells = bmesh.cells()

  vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
  vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

  normals = np.cross(vec1, vec2)
  normals /= np.sqrt((normals**2).sum(axis=1))[:, np.newaxis]

  # Ensure outward pointing normal
  #bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
  #normals[bmesh.cell_orientations() == 1] *= -1

  V = VectorFunctionSpace(bmesh, 'DG', 0)
  norm = Function(V)
  nv = norm.vector()

  for n in (0,1,2):
    dofmap = V.sub(n).dofmap()
    for i in xrange(dofmap.global_dimension()):
      dof_indices = dofmap.cell_dofs(i)
      assert len(dof_indices) == 1
      nv[dof_indices[0]] = normals[i, n]

  return norm

n     = 10
mesh  = UnitCubeMesh(n,n,n)
bmesh = BoundaryMesh(mesh, 'exterior')
#mesh = Mesh('meshes/unit_cube_mesh.xml')

# refine mesh :
origin = Point(0.0,0.5,0.5)
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
H = CellSize(mesh)
N = FacetNormal(mesh)
A = FacetArea(mesh)

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

File('output/n.pvd') << get_facet_normal(bmesh)

uv = u.vector().array()

N = as_vector([-1,0,0])

M  = assemble(w * v * dx)

s  = assemble(-dot(grad(u), N) * v * dx)
b  = assemble(l)
K  = assemble(a)
h  = project(H,Q).vector().array()/1.2

t  = 1/h**2*(np.dot(K.array(),uv) - b.array())

q = Function(Q, name='q')
q.vector().set_local(t)
q.vector().apply('insert')

File('output/q.pvd') << q

fx = Function(Q, name='fx')
solve(M, fx.vector(), s)
File('output/fx.pvd') << fx

fig = figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

plot_matrix(M.array(), ax1, r'mass matrix $M$',      continuous=True)
plot_matrix(K.array(), ax2, r'stiffness matrix $K$', continuous=True)

tight_layout()
show()



