import re
from collections.abc import Iterable

import bokeh.plotting as bp
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import layout
from bokeh.resources import INLINE
import numpy as np
import torch
import matplotlib       # for imshow palette
import matplotlib.cm as cm

__version__ = '0.1.5'

output_notebook(resources=INLINE)
#output_notebook()

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
BLACK = '#000000'
COLORS = {'b': BLUE, 'g': GREEN, 'o': ORANGE, 'r': RED, 'k': BLACK}

def figure(plot_width=900, plot_height=300, active_scroll='wheel_zoom', **kwargs):
    return bp.figure(plot_width=plot_width, plot_height=plot_height,
                     active_scroll=active_scroll, **kwargs)

def loglog_figure(plot_width=900, plot_height=300, active_scroll='wheel_zoom', **kwargs):
    return bp.figure(plot_width=plot_width, plot_height=plot_height, 
            active_scroll=active_scroll, 
            x_axis_type='log', y_axis_type='log', **kwargs)

#from itertools import zip_longest
#def grouper(iterable, n, fillvalue=None):
#    "Collect data into fixed-length chunks or blocks"
#    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
#    args = [iter(iterable)] * n
#    return zip_longest(*args, fillvalue=fillvalue)
#q = len(style)//n
#styles = [''.join(q) for q in grouper(style, q)]

def is_2d(y):
    return isinstance(y, torch.Tensor) and y.dim()==2 or \
      not isinstance(y, torch.Tensor) and isinstance(y, Iterable) and len(y) and isinstance(y[0], Iterable)

class ParseError(Exception):
    pass

def parse(x, y, style):
    tr = []
    if is_2d(y):
        n = len(y)
        if isinstance(style, (tuple, list)):
            styles = style
            if len(styles) != n:
                raise ParseError('len(styles)={len(styles)} does not match len(y)={len(y)}')
        else:
            line_style = re.sub('[a-z]', '', style) or '-'
            colors = re.sub('[^a-z]', '', style) or 'b'
            if len(colors) == n:
                styles = [line_style+c for c in colors]
            else:
                styles = [line_style+colors]*n
        if is_2d(x):
            for xi, yi, si in zip(x, y, styles):
                tr.append((xi, yi, si))
        else:
            for yi, si in zip(y, styles):
                if x is None:
                    xi = list(range(len(yi)))
                else:
                    xi = x
                tr.append((xi, yi, si))
    else:
        if x is None:
            x = list(range(len(y)))
        tr.append((x, y, style))
    return tr

def plot(*args, p=None, hover=False, mode='plot', **kwargs):
    try:
        show = p is None
        if show:
            if mode == 'plot':
                p = figure()
            elif mode == 'semilogx':
                p = figure(x_axis_type='log')
            elif mode == 'semilogy':
                p = figure(y_axis_type='log')
            elif mode == 'loglog':
                p = figure(x_axis_type='log', y_axis_type='log')
                print('ok')
        notebook_handle = kwargs.pop('notebook_handle', False)
        if hover:
            p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))
        tr = []
        style = '-'
        if len(args) in (1, 2):
            x = None
            if len(args) == 1:
                y = args[0]
            elif isinstance(args[1], str) or \
                 isinstance(args[1], (tuple, list)) and len(args[0]) > 0 and \
                 isinstance(args[1][0], str):
                y, style = args
            else:
                x, y = args
            tr.extend(parse(x, y, style))
        elif len(args) % 3 == 0:
            for h in range(len(args)//3):
                x, y, style = args[3*h:3*(h+1)]
                tr.extend(parse(x, y, style))
        base_color = kwargs.pop('color', BLUE)
        for x, y, style in tr:
            if isinstance(y, torch.Tensor):
                y = y.detach().numpy()
            source = ColumnDataSource(data=dict(x=x, y=y))
            if style and style[-1] in COLORS:
                color = COLORS[style[-1]]
                style = style[:-1]
            else:
                color = base_color
            if not style or '-' in style:
                p.line('x', 'y', source=source, color=color, **kwargs)
            if '.' in style:
                p.circle('x', 'y', source=source, color=color, **kwargs)
        handle = None
        if show:
            handle = bp.show(p, notebook_handle=notebook_handle)
        return source if notebook_handle else None
    except ParseError as e:
        print(e)

# eg:
# plot([1,2,3])
# plot(np.array([1,2,3]))
# plot(torch.tensor((1,2,3)))
# plot([1,2,3], '.-')
# plot(np.array([1,2,3]), '.-')
# plot(torch.tensor((1,2,3)), '.-')
# plot([(1,2,3), (3,2,1)])
# plot([(1,2,3), (3,2,1)], '.-')
# plot([np.array((3,2,1)), torch.tensor((1,2,3))])
# plot([np.array((3,2,1)), torch.tensor((1,2,3))], '.-')
# plot((1,2,3), [(1,4,9), (1,8,27)])  # two plots with same x values

def semilogx(*args, **kwargs):
    kwargs['mode'] = 'semilogx'
    plot(*args, **kwargs)

def semilogy(*args, **kwargs):
    kwargs['mode'] = 'semilogy'
    plot(*args, **kwargs)

def loglog(*args, **kwargs):
    kwargs['mode'] = 'loglog'
    plot(*args, **kwargs)

colormap =cm.get_cmap("viridis")
bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
def imshow(im):
    p = figure()
    p.image([im], x=[0], y=[0], dw=[im.shape[1]], dh=[im.shape[0]], palette=bokehpalette)
    bp.show(p)

def load_ipython_extension(ipython):
    ipython.user_ns.update(dict(figure=figure, 
        loglog_figure=loglog_figure,
        plot=plot, 
        show=bp.show,
        semilogx=semilogx, semilogy=semilogy, loglog=loglog,
        RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK, 
        push_notebook=push_notebook,
        bp=bp, imshow=imshow))
