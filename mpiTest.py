import numpy as np
from fenics import *
from mpi4py import MPI as nMPI
from scipy.interpolate import RectBivariateSpline
import resource

comm = nMPI.COMM_WORLD
name = nMPI.Get_processor_name()
size = comm.Get_size()
rank = comm.Get_rank()

n    = 10
mesh = UnitSquareMesh(n,n)
Q    = FunctionSpace(mesh, "CG", 1)

m    = 1000
data = np.zeros((m,m))

x = np.linspace(0,1,m)
y = np.linspace(0,1,m)
  
def get_spline_expression(data):
  spline = RectBivariateSpline(x, y, data)
  
  class newExpression(Expression):
    def __init__(self, element=None):
      pass
    def eval(self, values, x):
      values[0] = spline(x[0], x[1])

  return newExpression(element = Q.ufl_element())

if rank == 0:
  data = np.outer(np.sin(2*pi*x), np.sin(2*pi*y))
  print "Process", rank, "contains the data with norm", np.linalg.norm(data)
  comm.Send(data, dest=1)

if rank == 1:
  comm.Recv(data, source=0)
  print "Process", rank, "received the data with norm", np.linalg.norm(data)

expr = get_spline_expression(data)

usg = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print "Using %i KB" % usg 
