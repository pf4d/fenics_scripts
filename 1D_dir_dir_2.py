from pylab  import *
from fenics import *

mesh = IntervalMesh(1000,-1,1)
Q    = FunctionSpace(mesh, 'CG', 1)

u    = interpolate(Expression('pow(x[0], 6)'),   Q)
v    = interpolate(Expression('pow(x[0], 2)'),   Q)
dudv = interpolate(Expression('3*pow(x[0], 4)'), Q)

dudv_1 = u.dx(0) * 1/v.dx(0)


x    = mesh.coordinates()[:,0]
u_v  = u.vector().array()
v_v  = v.vector().array()
d_va = dudv.vector().array()
d_v1 = project(dudv_1).vector().array()

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

fig  = figure()
ax   = fig.add_subplot(111)

purp = '#880cbc'
grun = '#77f343'

ax.plot(x, u_v,  'k',   lw=2.0, label=r'$u$')
ax.plot(x, v_v,  'r',   lw=2.0, label=r'$v$')

ax.plot(x, d_va, color=grun, ls='-',  lw=2.0, 
        label=r'$\frac{du}{dv}$ - analytical')
ax.plot(x, d_v1, color=purp, ls='--', lw=2.0,
        label=r'$\frac{du}{dv}$ - numerical')

ax.grid()
ax.set_xlabel(r'$x$')
leg = ax.legend(loc='upper center')
leg.get_frame().set_alpha(0.5)
tight_layout()
show()




