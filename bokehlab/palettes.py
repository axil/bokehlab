import numpy as np
import matplotlib       # for imshow palette

def get_mpl(name):
    colormap = matplotlib.colormaps[name]
    palette = [matplotlib.colors.rgb2hex(m) 
                for m in colormap(np.arange(colormap.N))]
    return palette
