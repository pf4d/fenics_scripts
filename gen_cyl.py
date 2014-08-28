from MeshGenerator import MeshGenerator
from pylab         import *


#===============================================================================
# generate the contour :

x = linspace(-1.0, 1.0, 100)
y = linspace(-1.0, 1.0, 100)

X,Y = meshgrid(x,y)

S = 1 - sqrt(X**2 + Y**2)

m = MeshGenerator(x, y, 'mesh', 'meshes/')

m.create_contour(S, zero_cntr=1e-16, skip_pts=4)
m.eliminate_intersections(dist=10)
#m.plot_contour()
m.write_gmsh_contour(lc=0.1, boundary_extend=False)
m.extrude(h=1, n_layers=10)
m.add_edge_attractor(1)
#field, ifield, lcMin, lcMax, distMin, distMax
m.add_threshold(2, 1, 0.2, 0.2, 0, 0.5)
m.finish(4)



