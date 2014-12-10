from scipy.sparse            import spdiags
from pylab                   import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors       import from_levels_and_colors

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

def plot_matrix(M, ax, title, continuous=False, cmap='Greys'):
  """
  plot a matrix <M> with title <title> and a colorbar on subplot (axes object) 
  <ax>.
  """
  M    = array(M)
  M    = M.round(decimals=9)
  cmap = cm.get_cmap(cmap)
  if not continuous:
    unq  = unique(M)
    num  = len(unq)
  im      = ax.imshow(M, cmap=cmap, interpolation='None')
  divider = make_axes_locatable(ax)
  cax     = divider.append_axes("right", size="5%", pad=0.05)
  ax.set_title(title)
  ax.axis('off')
  cb = colorbar(im, cax=cax)
  if not continuous:
    cb.set_ticks(unq)
    cb.set_ticklabels(unq)



