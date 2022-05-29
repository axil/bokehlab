import sys
import re
from collections.abc import Iterable
from collections import deque
from itertools import cycle
from datetime import datetime

from IPython.core.magic import register_line_magic

#USE_TORCH = 0

import bokeh.plotting as bp
import bokeh.layouts as bl
from bokeh.models import HoverTool, ColumnDataSource, Span, CustomJSHover, DataTable, TableColumn, \
    DatetimeAxis
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import layout
from bokeh.resources import INLINE
import numpy as np
import pandas as pd
from jupyter_bokeh import BokehModel
import ipywidgets as ipw

#if USE_TORCH:
#    import torch
#else:
#    class torch:
#        class Tensor:
#            pass

import matplotlib       # for imshow palette
import matplotlib.cm as cm

#from .parser import parse

__version__ = '0.2.3'

output_notebook(resources=INLINE)
#output_notebook()

DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 300
DEFAULT_IMAGE_WIDTH = 300
DEFAULT_IMAGE_HEIGHT = 300

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
BLACK = '#000000'
COLORS = {'b': BLUE, 'g': GREEN, 'O': ORANGE, 'r': RED, 'k': BLACK}
def get_color(c):
    if c == 'a':
        return next(AUTOCOLOR[0])
    else:
        return COLORS.get(c, c)
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

def figure(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, active_scroll='wheel_zoom', **kwargs):
    return bp.figure(plot_width=width, plot_height=height,
                     active_scroll=active_scroll, **kwargs)

def loglog_figure(width=900, height=300, active_scroll='wheel_zoom', **kwargs):
    return bp.figure(plot_width=width, plot_height=height,
            active_scroll=active_scroll,
            x_axis_type='log', y_axis_type='log', **kwargs)

# ________________________________ parser __________________________________________

class ParseError(Exception):
    pass

class Missing:
    pass

def compare(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for q, r in zip(a, b):
            if not compare(q, r):
                return False
    elif isinstance(a, (np.ndarray, pd.Index, pd.Series)) or \
         isinstance(b, (np.ndarray, pd.Index, pd.Series)):
        a, b = np.array(a), np.array(b)
        if a.shape != b.shape or not np.allclose(a, b):   
            return False
    elif a != b:
        return False
    return True

def test(a, b):
    if compare(a, b):
        sys.stdout.write('.')
        sys.stdout.flush()
    else:
        import pdb; pdb.set_trace()
        print(f'{repr(a)} != {repr(b)}')

def is_string(arg):
    return isinstance(arg, str) or \
           isinstance(arg, (tuple, list)) and len(arg) > 0 and \
           isinstance(arg[0], str)

def choose(a, b, name):
    if a is not None and b is not None:
        raise ValueError(f'Ambiguous {name}: both positional and keyword argument')
    else:
        return a if a is not None else b

def broadcast_str(v, n, default, name):
    if v is None:
        v = [default] * n
    elif isinstance(v, str):
        v = [v] * n
    elif isinstance(v, (list, tuple)):
        if len(v) == 1:
            v *= n
        elif len(v) != n:
            raise ValueError(f'len({name})={len(v)} does not match len(y)={n}')
    return v

def broadcast_num(v, n, default, name):
    if v is None:
        v = [default] * n
    elif isinstance(v, (int, float)):
        v = [v] * n
    elif isinstance(v, (list, tuple)):
        if len(v) == 1:
            v *= n
        elif len(v) != n:
            raise ValueError(f'len({name})={len(v)} does not match len(y)={n}')
    return v
           
def parse(*args, style=None, color=None, label=None):
    _style = _color = _label = None
    
    x = Missing()
    if len(args) == 0:
        return []
    elif len(args) == 1:
        y = args[0]
    elif len(args) == 2:
        if is_string(args[1]):
            y, _style = args
        else:
            x, y = args
    elif len(args) == 3:
        if is_string(args[1]):
            y, _style, _color = args
        else:
            x, y, _style = args
    elif len(args) == 4:
        if is_string(args[1]):
            y, _style, _color, _label = args
        else:
            x, y, _style, _color = args
    elif len(args) == 5:
        x, y, _style, _color, _label = args
    
    style = choose(style, _style, 'style')
    color = choose(color, _color, 'color')
    label = choose(label, _label, 'label')
    
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = [y]
        elif y.ndim == 2:
            y = y.T
        else:
            raise ValueError(f'y is expected to be 1 or 2 dimensional, got {len(y.shape)} instead')

    elif isinstance(y, (list, tuple)):
        if len(y) == 0:         # plot([], [])
            pass
        else:
            if isinstance(y[0], (int, float)):      # flat list: [1,2,3]
                y = [y]
            elif isinstance(y[0], (list, tuple)):   
                if len(y[0]) > 0:
                    if isinstance(y[0][0], (int, float)):  # nested list: [[1,2,3],[4,5,6]]
                        pass
                    else:
                        raise TypeError(f'Unsupported y[0][0] type: {type(y[0][0])}')
            elif isinstance(y[0], np.ndarray):       # list of numpy arrays
                pass
            else:
                raise TypeError(f'Unsupported y[0] type: {type(y[0])}')

    elif isinstance(y, dict):
        if label is None:
            label = list(y.keys())
        y = list(y.values())
    
    elif isinstance(y, pd.Series):
        if isinstance(x, Missing):
            x = [y.index]
        y = [y.values]
    
    elif isinstance(y, pd.DataFrame):
        if isinstance(x, Missing):
            x = [y.index]
        if label is None:
            label = list(map(str, y.columns))
        y = [y[col].values for col in y.columns]

    # By this point, y is a list of n arrays, each corresponding to a separate plot

    n = len(y)    # number of plots
    
    if isinstance(x, Missing):
        x = [np.arange(len(yi)) for yi in y]

    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = [x] * n
        elif x.ndim == 2:
            if x.shape[1] != n:
                raise ValueError(f'Wrong number of columns in x: expected {n}, got {x.shape[1]}')
            elif any(x.shape[0] != len(yi) for yi in y):
                raise ValueError(f'Wrong number of rows in x: got {x.shape[0]}')
            else:
                x = x.T
        else:
            raise ValueError(f'x is expected to be 1 or 2-dimensional, got {x.ndim} dimensions')

    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            if len(y) != 0:
                raise ValueError('Length of x is 0 while len(y) is {len(y)}')
        elif isinstance(x[0], (int, float)):
            x = [x] * n
        elif isinstance(x[0], (list, tuple, np.ndarray, pd.Index)):
            if len(x) == 1:
                x *= n
            elif len(x) != n:
                raise ValueError(f'Number of x arrays = {len(x)} must either match number of y arrays = {n} or be equal to one')

    style = broadcast_str(style, n, '-', 'style')
    color = broadcast_str(color, n, 'a', 'color')
    
    if label is None:
        label = [None] * n
    elif isinstance(label, str) and n == 1:
        label = [label]
    elif isinstance(label, (list, tuple)) and len(label) == n:
        pass
    else:
        raise ValueError(f'length of label = {len(label)} must match the number of plots = {n}')

    return list(zip(x, y, style, color, label))

def test_parse_arr(x, x1, y, y1):
    x0 = [0, 1, 2]
    test(parse(y), [(x0, y, '-', 'a', None)])
    
    test(parse(y), [(x0, y, '-', 'a', None)])
    test(parse(y, '.-'), [(x0, y, '.-', 'a', None)])
    test(parse(y, style='.-'), [(x0, y, '.-', 'a', None)])
    test(parse(y, color='g'), [(x0, y, '-', 'g', None)])
    test(parse(y, '.-', 'g'), [(x0, y, '.-', 'g', None)])
    test(parse(y, '.-', 'g', label='y'), [(x0, y, '.-', 'g', 'y')])
    
    test(parse(x, y), [(x, y, '-', 'a', None)])
    test(parse(x, y, '.-'), [(x, y, '.-', 'a', None)])
    test(parse(x, y, '.-', 'g'), [(x, y, '.-', 'g', None)])
    test(parse(x, y, '.-', 'g', label='y'), [(x, y, '.-', 'g', 'y')])
    
    test(parse(x, y, style='.-'), [(x, y, '.-', 'a', None)])
    test(parse(x, y, color='g'), [(x, y, '-', 'g', None)])
    test(parse(x, y, label='y'), [(x, y, '-', 'a', 'y', None)])
    test(parse(x, y, style='.-', color='g'), [(x, y, '.-', 'g', None)])
    test(parse(x, y, style='.-', color='g', label='y'), [(x, y, '.-', 'g', 'y')])
    
    test(parse(x, [y, y1]), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)])
    test(parse(x, [y, y1], '.-'), [(x, y, '.-', 'a', None), (x, y1, '.-', 'a', None)])
    test(parse(x, [y, y1], ['.', '-']), [(x, y, '.', 'a', None), (x, y1, '-', 'a', None)])
    test(parse(x, [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)])
    test(parse(x, [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x, y1, '-', 'g', None)])
    test(parse(x, [y, y1], ['.', '-'], ['r', 'g'], label=['y1', 'y2']), [(x, y, '.', 'r', 'y1'), (x, y1, '-', 'g', 'y2')])
    
    test(parse([x, x1], [y, y1]), [(x, y, '-', 'a', None), (x1, y1, '-', 'a', None)])
    test(parse([x, x1], [y, y1], '.-'), [(x, y, '.-', 'a', None), (x1, y1, '.-', 'a', None)])
    test(parse([x, x1], [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x1, y1, '.-', 'g', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['b']), [(x, y, '.', 'b', None), (x1, y1, '-', 'b', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x1, y1, '-', 'g', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g'], label=['y1', 'y2']), [(x, y, '.', 'r', 'y1'), (x1, y1, '-', 'g', 'y2')])
    print()

def test_parse_arrays():
    x = [1, 2, 3]
    x1 = [-1, -2, -3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    test_parse_arr(x, x1, y, y1)
    test_parse_arr(x, x1, np.array(y), y1)
    test_parse_arr(np.array(x), x1, y, y1)
    test_parse_arr(np.array(x), x1, np.array(y), y1)
    test_parse_arr(np.array(x), np.array(x1), np.array(y), np.array(y1))

def test_parse_np():
    x0 = [0, 1, 2]
    x = [1, 2, 3]
    x1 = [-1, -2, -3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    xx = np.array([[1, 2, 3], [-1, -2, -3]]).T
    yy = np.array([[1, 4, 9], [-1, -4, -9]]).T
    test(parse(yy), [(x0, y, '-', 'a', None), (x0, y1, '-', 'a', None)])
    test(parse(x, yy), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)]), 
    test(parse(np.array(x), yy), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)]), 
    test(parse(xx, yy), [(x, y, '-', 'a', None), (x1, y1, '-', 'a', None)])
    test(parse(xx, yy, '.'), [(x, y, '.', 'a', None), (x1, y1, '.', 'a', None)])
    test(parse(xx, yy, '.-', 'g'), [(x, y, '.-', 'g', None), (x1, y1, '.-', 'g', None)])
    print()

def test_parse_pd():
    x0 = np.array([0, 1, 2])
    x = np.array([1, 2, 3])
    x1 = np.array([-1, -2, -3])
    x2 = np.array([-10, -20, -30])
    y = np.array([1, 4, 9])
    y1 = np.array([-1, -4, -9])
    df = pd.DataFrame({'y': y, 'y1': y1})
    df1 = pd.DataFrame({'y': y, 'y1': y1}, index=x)
    test(parse(df), [(x0, y, '-', 'a', 'y'), (x0, y1, '-', 'a', 'y1')])
    test(parse(df1), [(x, y, '-', 'a', 'y'), (x, y1, '-', 'a', 'y1')])
    test(parse(x1, df), [(x1, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    test(parse(x1, df1), [(x1, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    test(parse([x1, x2], df), [(x1, y, '-', 'a', 'y'), (x2, y1, '-', 'a', 'y1')])
    test(parse([x1, x2], df1), [(x1, y, '-', 'a', 'y'), (x2, y1, '-', 'a', 'y1')])
    test(parse(df, label=['Y', 'Y1']), [(x0, y, '-', 'a', 'Y'), (x0, y1, '-', 'a', 'Y1')])
    print()

def test_parse_dicts():
    x0 = np.array([0, 1, 2])
    x = np.array([1, 2, 3])
    x1 = np.array([-1, -2, -3])
    y = np.array([1, 4, 9])
    y1 = np.array([-1, -4, -9])
    d = {'y': y, 'y1': y1}
    test(parse(d), [(x0, y, '-', 'a', 'y', None), (x0, y1, '-', 'a', 'y1', None)])
    test(parse(x, d), [(x, y, '-', 'a', 'y', None), (x, y1, '-', 'a', 'y1', None)])
    test(parse([x, x1], d), [(x, y, '-', 'a', 'y', None), (x1, y1, '-', 'a', 'y1', None)])
    print()

# __________________________________________________________________________________

class BokehWidget(BokehModel):

    def on_change(self, *args, **kwargs):
        return self.model.on_change(*args, **kwargs)
    
    def on_event(self, *args, **kwargs):
        return self.model.on_event(*args, **kwargs)


def check_dt(quartuples):
    res = None
    for q in quartuples:
        if len(q[0]) == 0:
            continue
        v = isinstance(q[0][0], (datetime, np.datetime64))
        if res is None:
            res = v
        elif res != v:
            raise ValueError(f'Either all x arrays should be of datetime type or none at all')
    return res

def plot(*args, style=None, color=None, label=None, line_width=None, alpha=None,
         p=None, hover=False, mode='plot', 
         width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
         hline=None, vline=None, hline_color='pink', vline_color='pink', 
         xlabel=None, ylabel=None, legend_loc=None, grid=True,
         background_fill_color=None, x_range=None, y_range=None,
         #get_handle=False, get_source=False, 
         get_sh=False, get_ps=False, get_ws=False, show=True, **kwargs):
#    print('(plot) FIGURE =', FIGURE)
#    try:
    if len(args) == 0 and hline is None and vline is None:
        raise ValueError('Either positional arguments, or hline/vline are required')
    if len(args) > 5:
        raise ValueError('Too many positional arguments, can not be more than 5')
    #show = p is None
    quintuples = parse(*args, color=color, style=style, label=label)
    is_dt = check_dt(quintuples)
    if p is None:
        if not FIGURE:
            kw = {'width': width, 'height': height}#'x_axis_type': None, 'y_axis_type': None}
            if mode == 'plot':
                pass
            elif mode == 'semilogx':
                kw['x_axis_type'] = 'log'
            elif mode == 'semilogy':
                kw['y_axis_type'] = 'log'
            elif mode == 'loglog':
                kw['x_axis_type'] = kw['y_axis_type'] = 'log'
            if is_dt:
                if 'x_axis_type' in kw and kw['x_axis_type'] is not None:
                    raise ValueError('datetime x values is incompatible with "%s"' % mode)
                else:
                    kw['x_axis_type'] = 'datetime'
            if background_fill_color is not None:
                kw['background_fill_color'] = background_fill_color
            if x_range is not None:
                kw['x_range'] = x_range
            if y_range is not None:
                kw['y_range'] = y_range
            p = figure(**kw)
            if grid is False:
                p.xgrid.visible = False
                p.ygrid.visible = False
            FIGURE.append(p)
        else:
            p = FIGURE[0]
            if is_dt and not isinstance(p.xaxis[0], DatetimeAxis):
                raise ValueError('cannot plot datetime x values on a non-datetime x axis')
            elif not is_dt and isinstance(p.xaxis[0], DatetimeAxis):
                raise ValueError('cannot plot non-datetime x values on a datetime x axis')

    if hover:
        if is_dt:
            p.add_tools(HoverTool(tooltips=[('x', '@x{%F}'), ('y', '@y'), ('name', '$name')],
                        formatters={'@x': 'datetime'}))#, '@y': lat_custom}))
        else:
            p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))

    n = len(quintuples)

    line_widths = broadcast_num(line_width, n, None, 'line_width')
    alphas = broadcast_num(alpha, n, None, 'alpha')

    display_legend = False
    sources = []
    for (x, y, style, color_str, label_i), line_width_i, alpha_i in zip(quintuples, line_widths, alphas):
        color = get_color(color_str)
#        if isinstance(y, torch.Tensor):
#            y = y.detach().numpy()
        if len(x) != len(y):
            raise ValueError(f'len(x)={len(x)} is different from len(y)={len(y)}')
        if len(x) and isinstance(x[0], str):
            raise ValueError('plotting strings in x axis is not supported')
        source = ColumnDataSource(data=dict(x=x, y=y))
        label_already_set = False
        if label_i:
            display_legend = True
        if not style or '-' in style:
            kw = kwargs.copy()
            if legend_loc != 'hide' and label_i is not None:
                kw['legend_label'] = label_i
            if hover:
                kw['name'] = label_i
            if line_width_i:
                kw['line_width'] = line_width_i
            if alpha_i:
                kw['alpha'] = alpha_i
            p.line('x', 'y', source=source, color=color, **kw)
            label_already_set = True
        if '.' in style:
            kw = kwargs.copy()
            label_j = None if label_already_set else label_i
            if legend_loc != 'hide' and label_j is not None:
                kw['legend_label'] = label_j
            if hover:
                kw['name'] = label_i
            p.circle('x', 'y', source=source, color=color, **kw)
        sources.append(source)

    if isinstance(hline, (int, float)):
        hline = [hline]
    if isinstance(hline, (list, tuple)):
        for y in hline:
            span = Span(location=y, dimension='width', line_color=hline_color, 
                        line_width=1, level='overlay')
            p.renderers.append(span)
    elif hline is not None:
        raise TypeError(f'Unsupported type of hline: {type(hline)}')

    if isinstance(vline, (int, float)):
        vline = [vline]
    if isinstance(vline, (list, tuple)):
        for x in vline:
            span = Span(location=x, dimension='height', line_color=vline_color, 
                        line_width=1, level='overlay')
            p.renderers.append(span)
    elif vline is not None:
        raise TypeError(f'Unsupported type of vline: {type(vline)}')
    if legend_loc != 'hide' and display_legend:
        if label is not None:
            p.legend.click_policy="hide"
        if legend_loc is not None:
            p.legend.location = legend_loc
    if xlabel is not None:
        p.xaxis.axis_label = xlabel
    if ylabel is not None:
        p.yaxis.axis_label = ylabel
    handle = None
    if get_sh:
        handle = bp.show(p, notebook_handle=True)
        FIGURE.clear()
        return sources[0] if len(sources)==1 else sources, handle
#    elif get_source:
#        return source
    elif get_ps:
        FIGURE.clear()
        return p, sources[0] if len(sources)==1 else sources
    elif get_ws:
        FIGURE.clear()
        return BokehWidget(p), sources[0] if len(sources)==1 else sources
    elif show is False:
        FIGURE.clear()
        return p
    else:
        return None
#    except ParseError as e:
#        print(e)

# eg:
# plot(np.array([1,2,3]))
# plot([1,2,3])
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

def xlabel(label, p=None, **kw):
    if p is None:
        if not FIGURE:
            p = figure(**kw)
            FIGURE.append(p)
        else:
            p = FIGURE[0]
    p.xaxis.axis_label = label

def ylabel(label, p=None, **kw):
    if p is None:
        if not FIGURE:
            p = figure(**kw)
            FIGURE.append(p)
        else:
            p = FIGURE[0]
    p.yaxis.axis_label = label

def xylabels(xlabel, ylabel, p=None, **kw):
    if p is None:
        if not FIGURE:
            p = figure(**kw)
            FIGURE.append(p)
        else:
            p = FIGURE[0]
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

def hist(x, nbins=30, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, get_ws=False, **kw):
    hist, edges = np.histogram(x, density=True, bins=nbins)
    p = figure(height=height, width=width)
    defaults = dict(fill_color="navy", line_color="white", alpha=0.5)
    defaults.update(kw)
    source = ColumnDataSource(data=dict(top=hist, bottom=np.zeros_like(hist), left=edges[:-1], right=edges[1:]))
    p.quad(source=source, **defaults)
    if get_ws:
        return BokehWidget(p), source
    else:
        bp.show(p)

def _ramp(cmap, padding):
    return imshow(np.arange(256)[None, :].T.repeat(30, axis=1), cmap=cmap, show=False, 
                  toolbar=False, grid=False, padding=padding)

def imshow(*ims, p=None, cmap='viridis', stretch=True, axes=False, toolbar=True, 
           width=None, height=None, 
           grid=True, flipud=False, hover=False, padding=0.1, 
           merge_tools=True, link=True, toolbar_location='right', show_cmap=False, # multiple image related
           get_ws=False, notebook_handle=False, show=True):     # i/o 
    if len(ims) > 1:
        ps = [imshow(im, cmap=cmap, stretch=stretch, axes=axes, 
                     toolbar=False if merge_tools else toolbar, 
                     width=width, height=height, grid=grid, flipud=flipud, 
                     hover=hover, padding=padding, show=False) 
                for i,im in enumerate(ims)]
        if link:
            for pi in ps[1:]:
                pi.x_range = ps[0].x_range
                pi.y_range = ps[0].y_range
        grid = bl.gridplot([ps], merge_tools=merge_tools, toolbar_location=toolbar_location)
        if show_cmap:
            grid = bl.row(grid, _ramp(cmap=cmap, padding=padding))
        if show:
            bp.show(grid)
            return
        else:
            return grid

    if isinstance(ims[0], (list, tuple)):
        ims = ims[0]
        if not isinstance(ims[0], (list, tuple)):
            ims = [ims]
        ps = []
        for i, ims_row in enumerate(ims):
            ps_row = [imshow(im, cmap=cmap, stretch=stretch, axes=axes, toolbar=toolbar, 
                      width=width, flipud=flipud, hover=hover, padding=padding, show=False) 
                      for i,im in enumerate(ims_row)]
            if link:
                p0 = ps_row[0]
                for pi in ps_row[1:]:
                    pi.x_range = p0.x_range
                    pi.y_range = p0.y_range
            if show_cmap:
                ps.append(bl.row(
                    bl.gridplot([ps_row], merge_tools=merge_tools, toolbar_location=toolbar_location),
                    _ramp(cmap=cmap, padding=padding),
                ))
            else:
                ps.append(bl.gridplot([ps_row], merge_tools=merge_tools, toolbar_location=toolbar_location))
        return bp.show(bl.column(ps))

    im = ims[0]
    if p is None:
        kw = {}
        if hover is True:
            kw['tooltips'] = [("x", "$x"), ("y", "$y"), ("value", "@image")]
        if width is None and height is None:    # by default, fix height, calculate widths
            height = DEFAULT_IMAGE_HEIGHT
        if width is None:        # calculate width from height, keeping aspect ratio
            if toolbar:
                width = int(height/im.shape[0]*im.shape[1])+30
            else:
                width = int(height/im.shape[0]*im.shape[1])
        if height is None:       # calculate height from width, keeping aspect ratio
            if toolbar:
                height = int((width-30)/im.shape[1]*im.shape[0])
            else:
                height = int(width/im.shape[1]*im.shape[0])
#        if not flipud:                 this does not work due to an issue in bokeh
#            p.y_range.flipped=True     workaround below 
        y_pad = im.shape[1]*padding/2 if padding is not None else 0
        if not flipud:
            kw['y_range'] = [im.shape[0]+y_pad, -y_pad]
        else:
            kw['y_range'] = [-y_pad, im.shape[0]+y_pad]
        x_pad = im.shape[0]*padding/2 if padding is not None else 0  # just for symmetry with y
        kw['x_range'] = [-x_pad, im.shape[1]+x_pad]
        p = figure(width, height, **kw)   

#    if padding is not None:            can be uncommented once the issue is resolved
#        # p.x_range.range_padding = p.y_range.range_padding = padding
#        p.x_range.range_padding = padding
    
    if grid is False:
        p.xgrid.visible = False
        p.ygrid.visible = False
    
    
    if axes is False:
        p.axis.visible=False
    if toolbar is False:
        p.toolbar.logo = None
        p.toolbar_location = None
    if np.issubdtype(im.dtype, np.floating):
        if stretch:
            _min, _max = im.min(), im.max()
            if _min == _max:
                if _min > 1.:
                    im = np.ones_like(im, dtype=np.uint8)
                elif _min < 0.:
                    im = np.zeros_like(im, dtype=np.uint8)
            else:
                im = (im-_min)/(_max-_min)
        im = (im*255).astype(np.uint8)
    elif im.dtype == np.uint8:
        pass
    elif np.issubdtype(im.dtype, np.integer):
        if stretch:
            _min, _max = im.min(), im.max()
            if _min == _max:
                im = np.zeros_like(im, dtype=np.uint8)
            else:
                im = ((im-_min)/(_max-_min)*255).astype(np.uint8)
    im = im.squeeze()
    if not flipud:
        im = np.flipud(im)
    if im.ndim in (2, 3):
        if im.ndim == 2:
            colormap = cm.get_cmap(cmap)
            palette = [matplotlib.colors.rgb2hex(m) 
                       for m in colormap(np.arange(colormap.N))]
        else:
            if im.shape[-1] == 3: # 3 is rgb; 4 means rgba already
                im = np.dstack([im, np.full_like(im[:,:,0], 255)])
            im = im.view(dtype=np.uint32).reshape(im.shape[:2])
        kw = dict(image=[im], x=[0], dw=[im.shape[1]])
        if not flipud:
            kw.update(dict(y=[im.shape[0]], dh=[im.shape[0]]))
        else:
            kw.update(dict(y=[0], dh=[im.shape[0]]))
        source = ColumnDataSource(data=kw)
        if im.ndim == 2:
            h = p.image(source=source, palette=palette)
        else:
            h = p.image_rgba(source=source)
    else:
        raise ValueError('Unsupported image shape: ' + str(im.shape))
    if get_ws:
        return BokehWidget(p), source
    if show:
        if show_cmap:
            bp.show(bl.row(p, _ramp(cmap=cmap)))
        else:
            bp.show(p, notebook_handle=notebook_handle)
    else:
        return p
    if notebook_handle:
        return h

def show_df(df, get_ws=False):
    source = ColumnDataSource({str(k): v for k, v in df.items()})
    columns = [
        TableColumn(field=str(q), title=str(q))
            for q in df.columns
    ] 
    data_table = DataTable(source=source, columns=columns, width=960)#, height=280)

    if get_ws:
        return BokehWidget(data_table), source
    else:
        bp.show(data_table)

def hstack(*args, show=True):
    all_bokeh = all(isinstance(arg, bl.LayoutDOM) for arg in args)
    if all_bokeh:
        p = bl.row(*args)
        if show:
            bp.show(p)
        else:
            return p
    else:
        converted = [BokehWidget(arg) if isinstance(arg, bl.LayoutDOM) else arg for arg in args]
        return ipw.HBox(converted)

def vstack(*args, show=True):
    all_bokeh = all(isinstance(arg, bl.LayoutDOM) for arg in args)
    if all_bokeh:
        p = bl.column(*args)
        if show:
            bp.show(p)
        else:
            return p
    else:
        converted = [BokehWidget(arg) if isinstance(arg, bl.LayoutDOM) else arg for arg in args]
        return ipw.VBox(converted)

class AutoShow(object):
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

def register_callbacks(ip):
    # Avoid re-registering when reloading the extension
    def register(event, function):
        for f in ip.events.callbacks[event]:
            if hasattr(f, 'bokeh_plot_method'):
                ip.events.unregister(event, f)
#                print('unregistered')
        ip.events.register(event, function)

    vw = AutoShow()
#    ip.events.register('pre_execute', vw.pre_execute)
    register('pre_run_cell', vw.pre_run_cell)
#    ip.events.register('post_execute', vw.post_execute)
    register('post_run_cell', vw.post_run_cell)

def load_ipython_extension(ip):
    register_callbacks(ip)
    ip.user_ns.update(dict(figure=figure,
        loglog_figure=loglog_figure,
        plot=plot,
        show=bp.show,
        semilogx=semilogx, semilogy=semilogy, loglog=loglog,
        xlabel=xlabel, ylabel=ylabel, xylabels=xylabels,
        RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK,
        push_notebook=push_notebook, BokehWidget=BokehWidget,
        bp=bp, bl=bl, imshow=imshow, hist=hist, show_df=show_df,
        hstack=hstack, vstack=vstack))


def test_exceptions_1():
    FIGURE.clear()
    AUTOCOLOR.clear()
    AUTOCOLOR.append(cycle(AUTOCOLOR_PALETTE))
    import pytest
    with pytest.raises(ValueError):
        plot([[1,2,3],[4,5,6]], [[1,2,3],[4,5,6],[7,8,9]])
    with pytest.raises(ValueError, match='length of label = 1 must match the number of plots = 2'):
        plot([[1,2,3],[4,5,6]], label='y')
    with pytest.raises(ValueError, match=re.escape('len(alpha)=2 does not match len(y)=1')):
        plot([1,2,3], [-1,-2,-3], alpha=[0.1, 0.7])

def test_exceptions():
    import pytest
    pytest.main(['-o', 'python_files=__init__.py', __file__, '-k', 'test_exceptions_'])

if __name__ == '__main__':
    test_parse_arrays()
    test_parse_np()
    test_parse_pd()
    test_parse_dicts()
    test_exceptions()

elif __name__ == 'bokehlab':
    import IPython
    ip = IPython.core.interactiveshell.InteractiveShell.instance()
    register_callbacks(ip)
