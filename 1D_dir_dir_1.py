from pylab  import *
from fenics import *

xmin = 0
xmax = 2*pi
mesh = IntervalMesh(1000, xmin, xmax)
Q    = FunctionSpace(mesh, 'CG', 1)

u    = interpolate(Expression('sin(x[0])'),  Q)
v    = interpolate(Expression('cos(x[0])'),  Q)
dudv = interpolate(Expression('-cos(x[0])/sin(x[0])'), Q)

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

ax.plot(x, u_v,  'k',   lw=2.0, label=r'$u(x) = \sin(x)$')
ax.plot(x, v_v,  'r',   lw=2.0, label=r'$v(x) = \cos(x)$')

ax.plot(x, d_va, color=grun, ls='-',  lw=2.0, 
        label=r'$\frac{du}{dv} = -\cot(x)$')
ax.plot(x, d_v1, color=purp, ls='--', lw=2.0,
        label=r'$\frac{du}{dv}$ - numerical')

ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylim([-3,3])
ax.set_xlim([xmin, xmax])
leg = ax.legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
tight_layout()
show()




