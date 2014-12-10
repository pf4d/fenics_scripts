from MeshGenerator import MeshGenerator
from pylab         import *


#===============================================================================
# generate the contour :

m = MeshGenerator(None, None, 'mesh', 'meshes/')
lc = 0.05

m.set_contour(array([[0,0],[0,1],[1,1],[1,0]]))
m.write_gmsh_contour(lc=lc, boundary_extend=False)
m.extrude(h=1, n_layers=10)
m.add_edge_attractor(1)
#field, ifield, lcMin, lcMax, distMin, distMax
m.add_threshold(2, 1, lc, lc, 0, 0.5)
m.finish(4)



