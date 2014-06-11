from dolfin import *
import numpy as np

n    = 3
mesh = BoxMesh(-1, -1, 0, 1, 1, 2, n, n, n)

# define function space :
Q = FunctionSpace(mesh, "CG", 2)

# function to be evaluated :
f = Expression('sqrt(pow(x[0],2) + pow(x[1], 2) + pow(x[2], 2))')
f = interpolate(f, Q)

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

n1 = np.sqrt(sum(norm_U**2))
n2 = np.sqrt(sum(norm_Uf.vector().array()**2))
print n1, '?=', n2



