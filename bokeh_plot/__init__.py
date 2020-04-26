import re
from collections.abc import Iterable
from collections import deque

USE_TORCH = 0

import bokeh.plotting as bp
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import layout
from bokeh.resources import INLINE
import numpy as np

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

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
FIGURE = []
AUTOCOLOR = deque()
REGISTERED = {}

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

class Missing:
    pass

def parse(x, y, style):
    tr = []
    if is_2d(y):
        n = len(y)
        if isinstance(style, (tuple, list)):
            styles = style
            if len(styles) != n:
                raise ParseError(f'len(styles)={len(styles)} does not match len(y)={len(y)}')
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
                if isinstance(x, Missing):
                    xi = list(range(len(yi)))
                else:
                    xi = x
                tr.append((xi, yi, si))
    else:
        if isinstance(x, Missing):
            x = list(range(len(y)))
        tr.append((x, y, style))
    return tr

def plot(*args, p=None, hover=False, mode='plot', hline=None, vline=None, **kwargs):
#    print('(plot) FIGURE =', FIGURE)
    try:
        #show = p is None
        if p is None:
            if not FIGURE:
                if mode == 'plot':
                    p = figure()
                elif mode == 'semilogx':
                    p = figure(x_axis_type='log')
                elif mode == 'semilogy':
                    p = figure(y_axis_type='log')
                elif mode == 'loglog':
                    p = figure(x_axis_type='log', y_axis_type='log')
                    print('ok')
                FIGURE.append(p)
#                print('A', FIGURE)
            else:
                p = FIGURE[0]
#                print('B')

        notebook_handle = kwargs.pop('notebook_handle', False)
        if hover:
            p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))
        tr = []
        style = '-'
        if len(args) in (1, 2):
            x = Missing()
            if len(args) == 1:
                y = args[0]
            elif isinstance(args[1], str) or \
                 isinstance(args[1], (tuple, list)) and len(args[0]) > 0 and \
                 isinstance(args[1][0], str):
                y, style = args
            else:
                x, y = args
            if isinstance(y, dict):
                x, y = list(y.keys()), list(y.values())
            tr.extend(parse(x, y, style))
        elif len(args) % 3 == 0:
            for h in range(len(args)//3):
                x, y, style = args[3*h:3*(h+1)]
                tr.extend(parse(x, y, style))
        base_color = kwargs.pop('color', BLUE)
        for x, y, style in tr:
            if style and style[-1] in COLORS:
                color = COLORS[style[-1]]
                style = style[:-1]
            else:
                color = COLORS[AUTOCOLOR.popleft()]
            if isinstance(y, torch.Tensor):
                y = y.detach().numpy()
            if isinstance(y, dict):
                x, y = list(y.keys()), list(y.values())
            if len(x) != len(y):
                raise ValueError(f'len(x)={len(x)} is different from len(y)={len(y)}')
            source = ColumnDataSource(data=dict(x=x, y=y))
            if not style or '-' in style:
                p.line('x', 'y', source=source, color=color, **kwargs)
            if '.' in style:
                p.circle('x', 'y', source=source, color=color, **kwargs)
        if isinstance(hline, (int, float)):
            span = Span(location=hline, dimension='width', line_color=color, line_width=1, level='overlay')
            p.renderers.append(span)
        elif isinstance(vline, (int, float)):
            span = Span(location=vline, dimension='height', line_color=color, line_width=1, level='overlay')
            p.renderers.append(span)
#        handle = None
#        if show:
#            handle = bp.show(p, notebook_handle=notebook_handle)
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
# * Errors:
# plot([1,2], [1,2,3]) => ValueError
# plot({1: 1, 2: 4, 3: 9}) => parabola

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

class VarWatcher(object):
#    def __init__(self, ip):
#        self.shell = ip
#        self.last_x = None
#        self.figures = []
#
#    def pre_execute(self):
#        self.last_x = self.shell.user_ns.get('x', None)
#
    def pre_run_cell(self, info):
        FIGURE.clear()
        AUTOCOLOR.clear()
        AUTOCOLOR.extend('bgork')
    pre_run_cell.bokeh_plot_method = True
#        print('pre_run Cell code: "%s"' % info.raw_cell)

#    def post_execute(self):
#        if self.shell.user_ns.get('x', None) != self.last_x:
#            print("x changed!")
#
    def post_run_cell(self, result):
#        print('Cell code: "%s"' % result.info.raw_cell)
        if result.error_before_exec:
            print('Error before execution: %s' % result.error_before_exec)
        else:
#            p = self.shell.user_ns.get('FIGURE', [])
#            print('(post_run) FIGURE=', FIGURE)
            if FIGURE:
                bp.show(FIGURE[0])
    post_run_cell.bokeh_plot_method = True

def load_ipython_extension(ip):
    
    # Avoid re-registering when reloading the extension
    def register(event, function):
        for f in ip.events.callbacks[event]:
            if hasattr(f, 'bokeh_plot_method'):
                ip.events.unregister(event, f)
#                print('unregistered')
        ip.events.register(event, function)

    vw = VarWatcher()
#    ip.events.register('pre_execute', vw.pre_execute)
    register('pre_run_cell', vw.pre_run_cell)
#    ip.events.register('post_execute', vw.post_execute)
    register('post_run_cell', vw.post_run_cell)
    ip.user_ns.update(dict(figure=figure,
        loglog_figure=loglog_figure,
        plot=plot,
        show=bp.show,
        semilogx=semilogx, semilogy=semilogy, loglog=loglog,
        RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK,
        push_notebook=push_notebook,
        bp=bp, imshow=imshow))
