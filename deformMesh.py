from fenics import *

xmin = 0
xmax = 40000
ymin = 0
ymax = 40000
nx   = 10
ny   = 10
nz   = 10

mesh      = UnitCubeMesh(nx,ny,nz)
#flat_mesh = Mesh(mesh)                       # this works
flat_mesh = UnitCubeMesh(nx,ny,nz)            # this fails
Q         = FunctionSpace(mesh, "CG", 1)

# width and origin of the domain for deforming x coord :
width_x  = xmax - xmin
offset_x = xmin

# width and origin of the domain for deforming y coord :
width_y  = ymax - ymin
offset_y = ymin

# Deform the square to the defined geometry :
for x,x0 in zip(mesh.coordinates(), flat_mesh.coordinates()):
  # transform x :
  x[0]  = x[0]  * width_x + offset_x
  x0[0] = x0[0]  * width_x + offset_x

  # transform y :
  x[1]  = x[1]  * width_y + offset_y
  x0[1] = x0[1]  * width_y + offset_y


File('output/mesh.pvd') << Function(Q)
