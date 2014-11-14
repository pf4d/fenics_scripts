from dolfin import *
import numpy as np

n    = 20
mesh = BoxMesh(-1, -1, 0, 1, 1, 2, n, n, n)

# define function space :
Q = FunctionSpace(mesh, "CG", 1)

# function to be evaluated :
f = Expression('sqrt(pow(x[0],2) + pow(x[1], 2) + pow(x[2], 2))')
f = interpolate(f, Q)

# analytical directional derivative :
g = Expression('x[2] / sqrt(pow(x[0],2) + pow(x[1],2) + pow(x[2],2) + eps)',
               eps=DOLFIN_EPS)
g = interpolate(g, Q)

# formulate variables :
gradf   = grad(f)
u,v,w   = gradf

u       = project(u)
v       = project(v)
w       = project(w)

u_v     = u.vector().array()
v_v     = v.vector().array()
w_v     = w.vector().array()

norm_U  = np.sqrt(u_v**2 + v_v**2 + w_v**2)
norm_Uf = project(sqrt(inner(gradf, gradf)), Q)

u_v_n   = u_v / norm_U
v_v_n   = v_v / norm_U
w_v_n   = w_v / norm_U

u.vector().set_local(u_v_n)
v.vector().set_local(v_v_n)
w.vector().set_local(w_v_n)

U       = as_vector([u,v,w])
d       = as_vector([0,0,1])
t       = project(dot(gradf, d))

coord = mesh.coordinates()[-1,:]
print t(coord), '?=', g(coord)

n1 = np.sqrt(sum(norm_U**2))
n2 = np.sqrt(sum(norm_Uf.vector().array()**2))
print n1, '?=', n2

File('output/f.pvd')     << f
File('output/g.pvd')     << g
File('output/t.pvd')     << t
#File('output/gradf.pvd') << project(gradf)

#plot(project(gradf))
#plot(project(U))

#plot(gradf)
#plot(t)
#interactive()



