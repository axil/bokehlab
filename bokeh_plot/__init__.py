import re
from collections.abc import Iterable
from collections import deque
from itertools import cycle

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

#from .parser import parse

__version__ = '0.1.8'

output_notebook(resources=INLINE)
#output_notebook()

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
BLACK = '#000000'
COLORS = {'b': BLUE, 'g': GREEN, 'o': ORANGE, 'r': RED, 'k': BLACK}
def get_color(c):
    if c == 'a':
        return next(AUTOCOLOR[0])
    else:
        return COLORS[c]
FIGURE = []
AUTOCOLOR = []
#AUTOCOLOR_PALETTE = [
#        "#1f77b4",    # b
#        "#2ca02c",    # g
#        "#ffbb78",    # o
#        "#d62728",    # r
#        "#9467bd",
#        "#98df8a",
#        "#ff7f0e",
#        "#ff9896",
##        "#c5b0d5",
#        "#8c564b",
#        "#c49c94",
#        "#e377c2",
#        "#f7b6d2",
#        "#7f7f7f",
#        "#bcbd22",
#        "#dbdb8d",
#        "#17becf",
#        "#9edae5"
#]
I20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]    # AaBb, interleaved
I19 = I20.copy()
del I19[9]                      # removing c5d0d5

C20 = I20[::2] + I20[1::2]      # ABab, consecutive

C19 = C20.copy()
del C19[14]                     # removing c5d0d5

C10 = C20[:10]                  # AB, consecutive

AUTOCOLOR_PALETTE = C19         # BOGR..bogr..
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

# ________________________________ parser __________________________________________

import re
from collections.abc import Iterable

USE_TORCH = 0

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

def is_2d(y):
    return isinstance(y, torch.Tensor) and y.dim()==2 or \
       not isinstance(y, torch.Tensor) and isinstance(y, Iterable) and len(y) and isinstance(y[0], Iterable)

class ParseError(Exception):
    pass

class Missing:
    pass

def parse_spec(spec):
    if spec is None:
        spec = ''
    style = re.sub('[a-z]', '', spec) or '-'
    color = re.sub('[^a-z]', '', spec) or 'a'
    return style, color

def parse3(x, y, spec): # -> list of (x, y, spec)
    tr = []
    #import ipdb; ipdb.set_trace()
    if is_2d(y):
        h, w = len(y), len(y[0])
        if isinstance(x, Missing) and w == 2 and h > 2:
            if isinstance(y, np.ndarray):
                tr.append((y[:,0], y[:,1], spec))
            else:
                _x, _y = [], []
                for row in y:
                    _x.append(row[0])
                    _y.append(row[1])
                tr.append((_x, _y, spec))
        else:
            n = len(y)
            if isinstance(spec, (tuple, list)):
                specs = spec
                if len(specs) != n:
                    raise ParseError(f'len(spec)={len(spec)} does not match len(y)={len(y)}')
            else:
                style, colors = parse_spec(spec)
                if len(colors) == n:
                    specs = [style+c for c in colors]
                else:
                    specs = [style+colors]*n
            if is_2d(x):
                for xi, yi, si in zip(x, y, specs):
                    tr.append((xi, yi, si))
            else:
                for yi, si in zip(y, specs):
                    if isinstance(x, Missing):
                        xi = list(range(len(yi)))
                    else:
                        xi = x
                    tr.append((xi, yi, si))
    else:
        if isinstance(x, Missing):
            x = list(range(len(y)))
        tr.append((x, y, spec))
    return tr

def test_parse3():
    x = [1,2,3]
    y = [1,4,9]
    y1 = [-1,-4,-9]
    assert parse3(x, y, '') == [(x, y, '')]
    assert parse3(x, [y, y1], '') == [(x, y, '-a'), (x, y1, '-a')]
    assert parse3(x, [y, y1], '.') == [(x, y, '.a'), (x, y1, '.a')]
    assert parse3(x, [y, y1], '.-') == [(x, y, '.-a'), (x, y1, '.-a')]
    assert parse3(x, [y, y1], 'gr') == [(x, y, '-g'), (x, y1, '-r')]
    assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g'), (x, y1, '.-r')]
    print(parse3(x, [y, y1], '.-gr'))

def parse(*args, color=None, legend=None):
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
        tr.extend(parse3(x, y, style))
    elif len(args) % 3 == 0:
        n = len(args)//3
        for h in range(n):
            x, y, style = args[3*h:3*(h+1)]
            tr.extend(parse3(x, y, style))
    n = len(tr)
    # color
    if isinstance(color, (list, tuple)):
        if len(color) != n:
            raise ValueError(f'len(color)={len(color)}; color should either be a string or a tuple of length {n}')
        colors = color
    elif isinstance(color, str):
        colors = [color] * n
    elif color is None:
        colors = 'a'*n
    else:
        raise ValueError(f'color={color}; it should either be a string or a tuple of length {n}')
    # legend
    if isinstance(legend, (list, tuple)):
        if len(legend) != n:
            raise ValueError(f'len(legend)={len(legend)}; legend should either be a string or a tuple of length {n}')
        legends = legend
    elif isinstance(legend, str):
        legends = [legend] * n
    elif legend is None:
        legends = [None] * n
    else:
        raise ValueError(f'legend={legend}; it should either be a string or a tuple of length {n}')
    qu = []
    for (x, y, spec), color, legend in zip(tr, colors, legends):
        style, _color = parse_spec(spec)
        if _color != 'a':
            color = _color
        qu.append((x, y, style, color, legend))
    return qu

def test_parser():
    x = [1,2,3]
    y = [1,4,9]
    y1 = [-1,-4,-9]
    #assert parse3(x, y, '') == [(x, y, '')]
    #assert parse3(x, [y, y1], '') == [(x, y, '-a'), (x, y1, '-a')]
    #assert parse3(x, [y, y1], '.') == [(x, y, '.a'), (x, y1, '.a')]
    #assert parse3(x, [y, y1], '.-') == [(x, y, '.-a'), (x, y1, '.-a')]
    #assert parse3(x, [y, y1], 'gr') == [(x, y, '-g'), (x, y1, '-r')]
    #assert parse3(x, [y, y1], '.-gr') == [(x, y, '.-g'), (x, y1, '.-r')]
    assert parse(y) == [([0, 1, 2], y, '-', 'a', None)]
    assert parse(x, y) == [(x, y, '-', 'a', None)]
    assert parse(x, y, '.') == [(x, y, '.', 'a', None)]
    assert parse(x, y, '.-') == [(x, y, '.-', 'a', None)]
    assert parse(x, y, '.-g') == [(x, y, '.-', 'g', None)]
    assert parse(x, y, '.-g', legend='aaa') == [(x, y, '.-', 'g', 'aaa')]
    assert parse(x, [y, y1], '.-', color=['r', 'g']) == [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)]
    assert parse(x, [y, y1], '.-rg', legend=['y', 'y1']) == [(x, y, '.-', 'r', 'y'), (x, y1, '.-', 'g', 'y1')]
    print(parse(x, [y, y1], '.-g', legend='aaa'))

# __________________________________________________________________________________

def plot(*args, p=None, hover=False, mode='plot', hline=None, vline=None, color=None,
         legend=None, legend_loc=None, **kwargs):
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
        quintuples = parse(*args, color=color, legend=legend)
        notebook_handle = kwargs.pop('notebook_handle', False)
        if hover:
            p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))
        for x, y, style, color_str, legend_i in quintuples:
            color = get_color(color_str)
            if isinstance(y, torch.Tensor):
                y = y.detach().numpy()
            if isinstance(y, dict):
                x, y = list(y.keys()), list(y.values())
            if len(x) != len(y):
                raise ValueError(f'len(x)={len(x)} is different from len(y)={len(y)}')
            source = ColumnDataSource(data=dict(x=x, y=y))
            legend_set = False
            if not style or '-' in style:
                p.line('x', 'y', source=source, color=color, legend_label=legend_i, **kwargs)
                legend_set = True
            if '.' in style:
                legend_j = None if legend_set else legend_i
                p.circle('x', 'y', source=source, color=color, legend_label=legend_j, **kwargs)
        if isinstance(hline, (int, float)):
            span = Span(location=hline, dimension='width', line_color=color, line_width=1, level='overlay')
            p.renderers.append(span)
        elif isinstance(vline, (int, float)):
            span = Span(location=vline, dimension='height', line_color=color, line_width=1, level='overlay')
            p.renderers.append(span)
        if legend is not None:
            p.legend.click_policy="hide"
        if legend_loc is not None:
            p.legend.location = legend_loc
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
# plot({1: 1, 2: 4, 3: 9}) => parabola
# plot(x, [sin(x), cos(x)])
# plot(x, [sin(x), cos(x)], 'gr')
# * Errors:
# plot([1,2], [1,2,3]) => ValueError

def semilogx(*args, **kwargs):
    kwargs['mode'] = 'semilogx'
    plot(*args, **kwargs)

def semilogy(*args, **kwargs):
    kwargs['mode'] = 'semilogy'
    plot(*args, **kwargs)

def loglog(*args, **kwargs):
    kwargs['mode'] = 'loglog'
    plot(*args, **kwargs)

def hist(x, bins=30):
    hist, edges = np.histogram(x, density=True, bins=bins)
    p = figure()
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="navy", line_color="white", alpha=0.5)
    bp.show(p)

colormap =cm.get_cmap("viridis")
bokehpalette = [matplotlib.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
def imshow(im, p=None, palette=None):
    if p is None:
        p = figure()
    if palette is None:
        palette = bokehpalette
    p.image([im], x=[0], y=[0], dw=[im.shape[1]], dh=[im.shape[0]], palette=palette)
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
        AUTOCOLOR.append(cycle(AUTOCOLOR_PALETTE))
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
        bp=bp, imshow=imshow, hist=hist))
