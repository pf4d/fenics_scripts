from pylab import *
import subprocess

class MeshGenerator(object):
  """
  generate a mesh.
  """ 
  def __init__(self, x, y, fn, direc):
    """
    Generate a mesh with DataInput object <dd>, output filename <fn>, and 
    output directory <direc>.
    """
    self.fn         = fn
    self.direc      = direc
    self.x, self.y  = x, y
    self.f          = open(direc + fn + '.geo', 'w')
    self.fieldList  = []  # list of field indexes created.
  
  def create_contour(self, var, zero_cntr, skip_pts):  
    """
    Create a contour of the data field with index <var> of <dd> provided at 
    initialization.  <zero_cntr> is the value of <var> to contour, <skip_pts>
    is the number of points to skip in the contour, needed to prevent overlap. 
    """
    # create contour :
    fig = figure()
    self.ax = fig.add_subplot(111)
    self.ax.set_aspect('equal')
    self.c = self.ax.contour(self.x, self.y, var, [zero_cntr])
    
    # Get longest contour:
    cl       = self.c.allsegs[0]
    ind      = 0
    amax     = 0
    amax_ind = 0
    
    for a in cl:
      if size(a) > amax:
        amax = size(a)
        amax_ind = ind
      ind += 1
    
    # remove skip points and last point to avoid overlap :
    longest_cont      = cl[amax_ind]
    self.longest_cont = longest_cont[::skip_pts,:][:-1,:]
    
  def set_contour(self,cont_array):
    """ This is an alternative to the create_contour method that allows you to 
    manually specify contour points.
    Inputs:
    cont_array : A numpy array of contour points (i.e. array([[1,2],[3,4],...])) 
    """
    self.longest_cont = cont_array
    
  def plot_contour(self):
    """
    Plot the contour created with the "create_contour" method.
    """
    ax = self.ax
    lc  = self.longest_cont
    ax.plot(lc[:,0], lc[:,1], 'r-', lw = 3.0)
    show()

  def eliminate_intersections(self, dist=10):
    """
    Eliminate intersecting boundary elements. <dist> is an integer specifiying 
    how far forward to look to eliminate intersections.
    """
    class Point:
      def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def ccw(A,B,C):
      return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
    
    def intersect(A,B,C,D):
      return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
  
    lc   = self.longest_cont 
    
    flag = ones(len(lc))
    for ii in range(len(lc)-1):
      
      A = Point(*lc[ii])
      B = Point(*lc[ii+1])
      
      for jj in range(ii, min(ii + dist, len(lc)-1)):
        
        C = Point(*lc[jj])
        D = Point(*lc[jj+1])
        
        if intersect(A,B,C,D) and ii!=jj+1 and ii+1!=jj:
          flag[ii+1] = 0
          flag[jj] = 0
    
    counter  = 0
    new_cont = zeros((sum(flag),2))
    for ii,fl in enumerate(flag):
      if fl:
        new_cont[counter,:] = lc[ii,:]
        counter += 1
    
    self.longest_cont = new_cont
  
  def restart(self):
    """
    clear all contents from the .geo file.
    """
    self.f.close
    self.f = open(self.direc + self.fn + '.geo', 'w') 
    print 'Reopened \"' + self.direc + self.fn + '.geo\".'
  
  def write_gmsh_contour(self, lc=100000, boundary_extend=True):  
    """
    write the contour created with create_contour to the .geo file with mesh
    spacing <lc>.  If <boundary_extend> is true, the spacing in the interior 
    of the domain will be the same as the distance between nodes on the contour.
    """ 
    #FIXME: sporadic results when used with ipython, does not stops writing the
    #       file after a certain point.  calling restart() then write again 
    #       results in correct .geo file written.  However, running the script 
    #       outside of ipython works.
    c   = self.longest_cont
    f   = self.f

    pts = size(c[:,0])

    # write the file to .geo file :
    f.write("// Mesh spacing\n")
    f.write("lc = " + str(lc) + ";\n\n")
    
    f.write("// Points\n")
    for i in range(pts):
      f.write("Point(" + str(i) + ") = {" + str(c[i,0]) + "," \
              + str(c[i,1]) + ",0,lc};\n")
    
    f.write("\n// Lines\n")
    for i in range(pts-1):
      f.write("Line(" + str(i) + ") = {" + str(i) + "," + str(i+1) + "};\n")
    f.write("Line(" + str(pts-1) + ") = {" + str(pts-1) + "," \
            + str(0) + "};\n\n")
    
    f.write("// Line loop\n")
    loop = ""
    loop += "{"
    for i in range(pts-1):
      loop += str(i) + ","
    loop += str(pts-1) + "}"
    f.write("Line Loop(" + str(pts+1) + ") = " + loop + ";\n\n")
    
    f.write("// Surface\n")
    surf_num = pts+2
    f.write("Plane Surface(" + str(surf_num) + ") = {" + str(pts+1) + "};\n\n")

    if not boundary_extend:
      f.write("Mesh.CharacteristicLengthExtendFromBoundary = 0;\n\n")

    self.surf_num = surf_num
    self.pts      = pts
    self.loop     = loop
  
  def extrude(self, h, n_layers):
    """
    Extrude the mesh <h> units with <n_layers> number of layers.
    """
    f = self.f
    s = str(self.surf_num)
    h = str(h)
    layers = str(n_layers)
    
    f.write("Extrude {0,0," + h + "}" \
            + "{Surface{" + s + "};" \
            + "Layers{" + layers + "};}\n\n")
  
  
  def add_box(self, field, vin, xmin, xmax, ymin, ymax, zmin, zmax): 
    """
    add a box to the mesh.  e.g. for Byrd Glacier data:
      
      add_box(10000, 260000, 620000, -1080000, -710100, 0, 0) 

    """ 
    f  = self.f
    fd = str(field)

    f.write("Field[" + fd + "]      =  Box;\n")
    f.write("Field[" + fd + "].VIn  =  " + float(vin)  + ";\n")
    f.write("Field[" + fd + "].VOut =  lc;\n")
    f.write("Field[" + fd + "].XMax =  " + float(xmax) + ";\n")
    f.write("Field[" + fd + "].XMin =  " + float(xmin) + ";\n")
    f.write("Field[" + fd + "].YMax =  " + float(ymax) + ";\n")
    f.write("Field[" + fd + "].YMin =  " + float(ymin) + ";\n")
    f.write("Field[" + fd + "].ZMax =  " + float(zmax) + ";\n")
    f.write("Field[" + fd + "].ZMin =  " + float(zmin) + ";\n\n")
    
    self.fieldList.append(field)

  def add_edge_attractor(self, field):
    """
    """
    fd = str(field)
    f  = self.f

    f.write("Field[" + fd + "]              = Attractor;\n")
    f.write("Field[" + fd + "].NodesList    = " + self.loop + ";\n")
    f.write("Field[" + fd + "].NNodesByEdge = 100;\n\n")

  def add_threshold(self, field, ifield, lcMin, lcMax, distMin, distMax):
    """
    """
    fd = str(field)
    f  = self.f

    f.write("Field[" + fd + "]         = Threshold;\n")
    f.write("Field[" + fd + "].IField  = " + str(ifield)  + ";\n")
    f.write("Field[" + fd + "].LcMin   = " + str(lcMin)   + ";\n")
    f.write("Field[" + fd + "].LcMax   = " + str(lcMax)   + ";\n")
    f.write("Field[" + fd + "].DistMin = " + str(distMin) + ";\n")
    f.write("Field[" + fd + "].DistMax = " + str(distMax) + ";\n\n")

    self.fieldList.append(field)
  
  def finish(self, field):
    """
    figure out background field and close the .geo file.
    """
    f     = self.f
    fd    = str(field)
    flist = self.fieldList

    # get a string of the fields list :
    l = ""
    for i,j in enumerate(flist):
      l += str(j)
      if i != len(flist) - 1:
        l += ', '
  
    # make the background mesh size the minimum of the fields : 
    if len(flist) > 0:
      f.write("Field[" + fd + "]            = Min;\n")
      f.write("Field[" + fd + "].FieldsList = {" + l + "};\n")
      f.write("Background Field    = " + fd + ";\n\n")
    else:
      f.write("Background Field = " + fd + ";\n\n")
    
    print 'finished, closing \"' + self.direc + self.fn + '.geo\".'
    f.close()
  
  def close_file(self):
    """
    close the .geo file down for further editing.
    """
    self.f.close()


  def create_2D_mesh(self):
    """
    create the 2D mesh to file <outfile>.msh.
    """
    cmd = 'gmsh ' + '-2 ' + self.direc + self.fn + '.geo'
    print "\nExecuting :\n\n\t", cmd, "\n\n"
    subprocess.call(cmd.split())


  def convert_msh_to_xml(self):
    """
    convert <mshfile> .msh file to .xml file <xmlfile> via dolfin-convert.
    """
    msh = self.direc + self.fn + '.msh'
    xml = self.direc + self.fn + '.xml'

    cmd = 'dolfin-convert ' + msh + ' ' + xml
    print "\nExecuting :\n\n\t", cmd, "\n\n"
    subprocess.call(cmd.split())

