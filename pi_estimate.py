from dolfin import *
from mshr   import Circle, generate_mesh

domain = Circle(Point(0.0,0.0), 1.0)

for res in [2**k for k in range(10)]:
	mesh   = generate_mesh(domain, res)
	A      = assemble(Constant(1) * dx(domain=mesh))
	print "resolution = %i, \t |A - pi| = %.5e" % (res, abs(A - pi))
