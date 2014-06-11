from dolfin import *

mesh = IntervalMesh(8, 0, 1)
Q    = FunctionSpace(mesh, 'CG', 1)
phi  = TestFunction(Q)
u    = TrialFunction(Q)

def left(x, on_boundary):
  tol = 1e-14
  return on_boundary and abs(x[0]) < tol

def right(x, on_boundary):
  tol = 1e-14
  return on_boundary and abs(x[0] - 1) < tol

gamma_l = DirichletBC(Q, 0.0, left)
gamma_r = DirichletBC(Q, 0.0, right)

k = inner(grad(phi), grad(u)) * dx
m = phi * u * dx

K = PETScMatrix()
M = PETScMatrix()

K = assemble(k, tensor=K)
M = assemble(m, tensor=M)

gamma_r.apply(K)
gamma_l.apply(K)
gamma_r.apply(M)
gamma_l.apply(M)

eigensolver = SLEPcEigenSolver(K,M)
eigensolver.solve()

print eigensolver.get_eigenvalue()

u = Function(Q)

#plot(mesh)
#interactive()
