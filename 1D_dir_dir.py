from pylab  import *
from fenics import *

def gauss(x):
  return exp(-(x/2)**2)

class F2(Expression):
  def eval(self, values, x): 
    values[0] = gauss(x[0])

mesh = IntervalMesh(1000,-pi,pi)
Q    = FunctionSpace(mesh, 'CG', 1)

f1   = Expression('sin(x[0])')
f2   = F2()
u    = interpolate(f1,Q)
v    = interpolate(f2,Q)

dudv = u.dx(0) * v

x    = mesh.coordinates()[:,0]

u_v  = u.vector().array()
v_v  = v.vector().array()
d_v  = project(dudv).vector().array()

fig  = figure()
ax   = fig.add_subplot(111)

ax.plot(x, u_v, 'k',   lw=2.0, label=r'$u$')
ax.plot(x, v_v, 'r',   lw=2.0, label=r'$v$')
ax.plot(x, d_v, 'k--', lw=2.0, label=r'$\frac{\partial u}{\partial v}$')

ax.grid()
ax.legend()
show()




