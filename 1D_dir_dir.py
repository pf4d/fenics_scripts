from pylab  import *
from fenics import *

mesh = IntervalMesh(1000,-pi,pi)
Q    = FunctionSpace(mesh, 'CG', 1)

y    = interpolate(Expression('pow(x[0], 6)'), Q)
t    = interpolate(Expression('pow(x[0], 2)'), Q)
d1   = interpolate(Expression('3*pow(x[0], 4)'), Q)

dydt = y.dx(0) * 1/t.dx(0)

x    = mesh.coordinates()[:,0]

y_v  = y.vector().array()
t_v  = t.vector().array()
d_1  = d1.vector().array()
d_2  = project(dydt).vector().array()

fig  = figure()
ax   = fig.add_subplot(111)

ax.plot(x, y_v, 'k',   lw=2.0, label=r'$y$')
ax.plot(x, t_v, 'r',   lw=2.0, label=r'$t$')
ax.plot(x, d_1, 'k--', lw=2.0, label=r'analytical $\frac{dy}{dt}$')
ax.plot(x, d_2, 'r--', lw=2.0, label=r'numerical $\frac{dy}{dt}$')

ax.grid()
ax.legend(loc='upper center')
show()




