import sys
src_directory = '../statistical_modeling'
sys.path.append(src_directory)

from src.regstats   import linRegstats, prbplotObj
from scipy.stats    import probplot
from dolfin         import *
from pylab          import *
from scipy          import random

mesh = UnitCubeMesh(10,10,10)          # original mesh
mesh.coordinates()[:,0] -= .5          # shift x-coords
mesh.coordinates()[:,1] -= .5          # shift y-coords
V    = FunctionSpace(mesh, "CG", 1)

# apply expression over cube for clearer results :
u_i  = Expression('sqrt(pow(x[0],2) + pow(x[1],2))')
v_i  = Expression('exp(-pow(x[0],2)/2 - pow(x[1],2)/2)')
z_i  = Expression('10 + 10 * sin(2*pi*x[0]) * sin(2*pi*x[1])')
x_i  = Expression('x[0]')
y_i  = Expression('x[1]')

u = interpolate(u_i, V)
v = interpolate(v_i, V)
z = interpolate(z_i, V)
x = interpolate(x_i, V)
y = interpolate(y_i, V)
w = project(u*v*z, V)

# apply some noise
w_v = w.vector().array()
w_v += 0.05*random.randn(len(w_v))
w.vector().set_local(w_v)
w.vector().apply('insert')

bmesh  = BoundaryMesh(mesh, "exterior")   # surface boundary mesh

cellmap = bmesh.entity_map(2)
pb      = CellFunction("size_t", bmesh, 0)
for c in cells(bmesh):
  if Facet(mesh, cellmap[c.index()]).normal().z() < 0:
    pb[c] = 1
submesh = SubMesh(bmesh, pb, 1)           # subset of surface mesh

Vs = FunctionSpace(submesh, "CG", 1)      # submesh function space

us  = Function(Vs)                        # desired function
vs  = Function(Vs)                        # desired function
zs  = Function(Vs)                        # desired function
ws  = Function(Vs)                        # desired function
xs  = Function(Vs)                        # x-coord
ys  = Function(Vs)                        # y-coord

us.interpolate(u)
vs.interpolate(v)
zs.interpolate(z)
xs.interpolate(x)
ys.interpolate(y)
ws.interpolate(w)

#File("output/us.pvd") << us
#File("output/vs.pvd") << vs
#File("output/u.pvd")  << u

u_v  = us.vector().array()
v_v  = vs.vector().array()
z_v  = zs.vector().array()
x_v  = xs.vector().array()
y_v  = ys.vector().array()
w_v  = ws.vector().array()

i    = argsort(u_v)
u_v  = u_v[i]
v_v  = v_v[i]
z_v  = z_v[i]
x_v  = x_v[i] 
y_v  = y_v[i] 
w_v  = w_v[i] 

x1   = u_v
x2   = v_v
x3   = z_v
x4   = np.sqrt(x_v**2 + y_v**2 + 1e-10)
x5   = x1 * x2 * x3

X    = array([x1, x2, x3, x4, x5])
yt   = w_v

out  = linRegstats(X, yt, 0.95)

bhat = out['bhat']
yhat = out['yhat']
ciy  = out['CIY']

print out['F_pval'], out['pval']

fig  = figure()
ax   = fig.add_subplot(111)

ax.plot(u_v, yt,     'ko', lw=2.0)
ax.plot(u_v, yhat,   'r-', lw=2.0)
#ax.plot(u_v, ciy[0], 'k:', lw=2.0)
#ax.plot(u_v, ciy[1], 'k:', lw=2.0)
ax.set_xlabel(r'$u$')
ax.set_ylabel(r'$w$')
grid()
show()



