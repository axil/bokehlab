import numpy as np
import matplotlib       # for imshow palette
import matplotlib.cm as cm

def get_mpl(name):
    colormap = matplotlib.colormaps[name]
    palette = [matplotlib.colors.rgb2hex(m) 
                for m in colormap(np.arange(colormap.N))]
    return palette
    

