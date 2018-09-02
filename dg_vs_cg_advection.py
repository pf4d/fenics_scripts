from dolfin     import *
from fenics_viz import *
from time       import time

class Inflow(SubDomain):
  def inside(self, x, on_boundary):
    return (x[1] < DOLFIN_EPS or x[0] < DOLFIN_EPS) \
           and on_boundary

cg_err_A  = []
dg_err_A  = []
cg_t_A    = []
dg_t_A    = []
cg_dim_A  = []
dg_dim_A  = []
n_a       = np.array([5,10,20,40,80,160])
o_a       = [1,2,3]

# polynomial order of the basis:
order = 2

# velocity function :
u_x = 1.0
u_y = 0.5
u   = Constant((u_x, u_y))

# source term :
f      = Constant(0.0)

# the linear differential operator for this problem (pure advection) :
def Lu(U): return dot(u, grad(U))

# exact solution :
phi_e = Expression('sin(5.0*pi*(x[1] - x[0]*u_y))', u_x=u_x, u_y=u_y, degree=2)

# Load mesh
for order in o_a:

  cg_err_a  = []
  dg_err_a  = []
  cg_t_a    = []
  dg_t_a    = []
  cg_dim_a  = []
  dg_dim_a  = []

  for n_dof in n_a:

    mesh = UnitSquareMesh(n_dof, n_dof, "crossed")

    # Defining the function spaces
    V_dg = FunctionSpace(mesh, "DG", order)
    V_cg = FunctionSpace(mesh, "CG", order)

    # mesh-related functions :
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x = SpatialCoordinate(mesh)

    # Test and trial functions
    v_dg   = TestFunction(V_dg)
    v_cg   = TestFunction(V_cg)
    phi_dg = TrialFunction(V_dg)
    phi_cg = TrialFunction(V_cg)

    # intrinsic time parameter :
    unorm  = sqrt(dot(u,u) + DOLFIN_EPS)
    tau    = h / (2 * unorm)

    # ( dot(v, n) + |dot(v, n)| )/2.0
    un = (dot(u, n) + abs(dot(u, n))) / 2.0

    # bilinear forms :
    a_dg = - Lu(v_dg) * phi_dg * dx \
           + dot(jump(un*phi_dg), jump(v_dg)) * dS \
           + dot(un, v_dg) * phi_dg * ds

    a_cg = + Lu(phi_cg) * v_cg * dx \
           + inner(Lu(v_cg), tau*Lu(phi_cg)) * dx

    # linear forms :
    L_dg = v_dg*f*dx

    L_cg = v_cg*f*dx + inner(Lu(v_cg), tau*f) * dx

    # set up boundary condition (apply strong BCs) :
    bc_dg = DirichletBC(V_dg, phi_e, Inflow(), "geometric")
    bc_cg = DirichletBC(V_cg, phi_e, Inflow(), "geometric")

    # solution function :
    phi_dg = Function(V_dg)
    phi_cg = Function(V_cg)

    # assemble, apply boundary conditions, and solve :
    t0_dg = time()
    A_dg  = assemble(a_dg)
    b_dg  = assemble(L_dg)
    bc_dg.apply(A_dg, b_dg)
    solve(A_dg, phi_dg.vector(), b_dg)
    tf_dg = time()

    t0_cg = time()
    A_cg = assemble(a_cg)
    b_cg = assemble(L_cg)
    bc_cg.apply(A_cg, b_cg)
    solve(A_cg, phi_cg.vector(), b_cg)
    tf_cg = time()

    # interpolate solution to the continuous function space :
    up = interpolate(phi_dg, V=V_cg)

    # calculate the exact solution :
    phi_exact_cg = interpolate(phi_e, V_cg)
    phi_exact_dg = interpolate(phi_e, V_dg)

    # Plot the result:
    dg_err = norm(phi_exact_dg.vector() - phi_dg.vector(), norm_type='linf')
    cg_err = norm(phi_exact_cg.vector() - phi_cg.vector(), norm_type='linf')
    print "for n = %i :\t DG error = %.4e,\t CG error = %.4e " \
          % (n_dof, dg_err, cg_err)

    dg_dim_a.append(V_dg.dim())
    cg_dim_a.append(V_cg.dim())
    dg_err_a.append(dg_err)
    cg_err_a.append(cg_err)
    cg_t_a.append(tf_cg - t0_cg)
    dg_t_a.append(tf_dg - t0_dg)

  dg_dim_A.append(np.array(dg_dim_a))
  cg_dim_A.append(np.array(cg_dim_a))
  dg_err_A.append(np.array(dg_err_a))
  cg_err_A.append(np.array(cg_err_a))
  cg_t_A.append(np.array(cg_t_a))
  dg_t_A.append(np.array(dg_t_a))

# Plot solution
#plot_variable(phi_exact_cg, 'phi_exact', './', plot_tp=False, show=False)
#plot_variable(phi_dg,       'phi_dg',    './', plot_tp=False, show=False)
#plot_variable(phi_cg,       'phi_cg',    './', plot_tp=False, show=False)

import matplotlib.pyplot as plt
import matplotlib        as mpl

mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'small'

fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ls_a = ['-', '--', ':']

for o, ls, cg_dim_a, cg_err_a, dg_dim_a, dg_err_a, cg_t_a, dg_t_a in \
    zip(o_a, ls_a, cg_dim_A, cg_err_A, dg_dim_A, dg_err_A, cg_t_A, dg_t_A):

  ax1.loglog(cg_dim_a, cg_err_a, c='r', ls=ls)
  ax1.loglog(dg_dim_a, dg_err_a, c='k', ls=ls)
  ax2.loglog(cg_dim_a, cg_t_a,   c='r', ls=ls, label=r"CG order = %i" % o)
  ax2.loglog(dg_dim_a, dg_t_a,   c='k', ls=ls, label=r"DG order = %i" % o)

ax1.set_ylabel(r"$\Vert u - u_{e} \Vert_{\infty}$")
ax1.set_xlabel(r"number of dofs")
ax1.grid()

ax2.legend()
ax2.set_ylabel('time to compute [s]')
ax2.set_xlabel('number of dofs')
ax2.grid()

plt.tight_layout()
plt.show()


