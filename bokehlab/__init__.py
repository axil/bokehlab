import sys
import re
from collections.abc import Iterable
from collections import deque
from itertools import cycle
from datetime import datetime

from IPython.core.magic import register_line_magic

USE_TORCH = 0

import bokeh.plotting as bp
import bokeh.layouts as bl
from bokeh.models import HoverTool, ColumnDataSource, Span, CustomJSHover, DataTable, TableColumn, \
    DatetimeAxis
from bokeh.io import output_notebook, push_notebook
from bokeh.layouts import layout
from bokeh.resources import INLINE
import numpy as np
import pandas as pd

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

import matplotlib       # for imshow palette
import matplotlib.cm as cm

#from .parser import parse

__version__ = '0.2.3'

output_notebook(resources=INLINE)
#output_notebook()

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

#USE_TORCH = 0
#
#if USE_TORCH:
#    import torch
#else:
#    class torch:
#        class Tensor:
#            pass

#def is_2d(y):
#    if isinstance(y, torch.Tensor):
#        return y.dim()==2
#    elif isinstance(y, np.ndarray):
#        return y.ndim==2
#    elif isinstance(y, pd.DataFrame):
#        return True
#    elif isinstance(y, pd.Series):
#        return False
#    else:
#        return isinstance(y, Iterable) and len(y) and \
#               isinstance(y[0], Iterable) and not isinstance(y[0], str)

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

def broadcast(v, n, default, name):
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
        if len(y) == 0:
            pass
        elif len(y) == 1:
            pass
        elif len(y) == 2:
            pass
        else:
            y = [y]

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
        elif len(x) == 1 and n != 1:
            x *= n
        elif len(x) == 2:
            pass
        else:
            x = [x]

    style = broadcast(style, n, '-', 'style')
    color = broadcast(color, n, 'a', 'color')
    label = broadcast(label, n, None, 'label')

    return list(zip(x, y, style, color, label))

def test_parse_arr(x, x1, y, y1):
    x0 = [0, 1, 2]
    test(parse(y), [(x0, y, '-', 'a', None)])
    
    test(parse(y), [(x0, y, '-', 'a', None)])
    test(parse(y, '.-'), [(x0, y, '.-', 'a', None)])
    test(parse(y, '.-'), [(x0, y, '.-', 'a', None)])
    test(parse(y, color='g'), [(x0, y, '-', 'g', None)])
    test(parse(y, label='y'), [(x0, y, '-', 'a', 'y')])
    test(parse(y, '.-', 'g'), [(x0, y, '.-', 'g', None)])
    test(parse(y, '.-', 'g', 'y'), [(x0, y, '.-', 'g', 'y')])
    
    test(parse(x, y), [(x, y, '-', 'a', None)])
    test(parse(x, y, '.-'), [(x, y, '.-', 'a', None)])
    test(parse(x, y, '.-', 'g'), [(x, y, '.-', 'g', None)])
    test(parse(x, y, '.-', 'g', 'y'), [(x, y, '.-', 'g', 'y')])
    
    test(parse(x, y, style='.-'), [(x, y, '.-', 'a', None)])
    test(parse(x, y, color='g'), [(x, y, '-', 'g', None)])
    test(parse(x, y, label='y'), [(x, y, '-', 'a', 'y')])
    test(parse(x, y, style='.-', color='g'), [(x, y, '.-', 'g', None)])
    test(parse(x, y, style='.-', color='g', label='y'), [(x, y, '.-', 'g', 'y')])
    
    test(parse(x, [y, y1]), [(x, y, '-', 'a', None), (x, y1, '-', 'a', None)])
    test(parse(x, [y, y1], '.-'), [(x, y, '.-', 'a', None), (x, y1, '.-', 'a', None)])
    test(parse(x, [y, y1], ['.', '-']), [(x, y, '.', 'a', None), (x, y1, '-', 'a', None)])
    test(parse(x, [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)])
    test(parse(x, [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x, y1, '-', 'g', None)])
    test(parse(x, [y, y1], ['.', '-'], ['r', 'g'], ['y', 'y1']), [(x, y, '.', 'r', 'y'), (x, y1, '-', 'g', 'y1')])
    
    test(parse([x, x1], [y, y1]), [(x, y, '-', 'a', None), (x1, y1, '-', 'a', None)])
    test(parse([x, x1], [y, y1], '.-'), [(x, y, '.-', 'a', None), (x1, y1, '.-', 'a', None)])
    test(parse([x, x1], [y, y1], '.-', ['r', 'g']), [(x, y, '.-', 'r', None), (x1, y1, '.-', 'g', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['b']), [(x, y, '.', 'b', None), (x1, y1, '-', 'b', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g']), [(x, y, '.', 'r', None), (x1, y1, '-', 'g', None)])
    test(parse([x, x1], [y, y1], ['.', '-'], ['r', 'g'], ['y', 'y1']), [(x, y, '.', 'r', 'y'), (x1, y1, '-', 'g', 'y1')])
    print()

def test_parse_arrs():
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
    test(parse(xx, yy, '.-', 'g', ['y', 'y1']), [(x, y, '.-', 'g', 'y'), (x1, y1, '.-', 'g', 'y1')])
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
    test(parse(d), [(x0, y, '-', 'a', 'y'), (x0, y1, '-', 'a', 'y1')])
    test(parse(x, d), [(x, y, '-', 'a', 'y'), (x, y1, '-', 'a', 'y1')])
    test(parse([x, x1], d), [(x, y, '-', 'a', 'y'), (x1, y1, '-', 'a', 'y1')])
    print()

# __________________________________________________________________________________

def check_dt(quintuples):
    res = None
    for q in quintuples:
        if len(q[0]) == 0:
            continue
        v = isinstance(q[0][0], (datetime, np.datetime64))
        if res is None:
            res = v
        elif res != v:
            raise ValueError(f'Either all x arrays should be of datetime type or none at all')
    return res


def plot(*args, style=None, color=None, label=None,
        p=None, hover=False, mode='plot', 
        plot_width=900, plot_height=300,
        hline=None, vline=None, hline_color='pink', vline_color='pink', 
        xlabel=None, ylabel=None, legend_loc=None, 
        notebook_handle=False, return_source=False, **kwargs):
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
            kw = {'plot_width': plot_width, 'plot_height': plot_height}#'x_axis_type': None, 'y_axis_type': None}
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
            p = figure(**kw)
            FIGURE.append(p)
#                print('A', FIGURE)
        else:
            p = FIGURE[0]
            if is_dt and not isinstance(p.xaxis[0], DatetimeAxis):
                raise ValueError('cannot plot datetime x values on a non-datetime x axis')
            elif not is_dt and isinstance(p.xaxis[0], DatetimeAxis):
                raise ValueError('cannot plot non-datetime x values on a datetime x axis')
#                print('B')

    if hover:
        if is_dt:
            p.add_tools(HoverTool(tooltips=[('x', '@x{%F}'), ('y', '@y'), ('name', '$name')],
                        formatters={'@x': 'datetime'}))#, '@y': lat_custom}))
        else:
            p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))
    for x, y, style, color_str, label_i in quintuples:
        color = get_color(color_str)
        if isinstance(y, torch.Tensor):
            y = y.detach().numpy()
        if isinstance(y, dict):
            x, y = list(y.keys()), list(y.values())
        if len(x) != len(y):
            raise ValueError(f'len(x)={len(x)} is different from len(y)={len(y)}')
        if len(x) and isinstance(x[0], str):
            raise ValueError('plotting strings in x axis is not supported')
        source = ColumnDataSource(data=dict(x=x, y=y))
        label_set = False
        if not style or '-' in style:
            kw = kwargs.copy()
            if legend_loc != 'hide' and label_i is not None:
                kw['legend_label'] = label_i
            if hover:
                kw['name'] = label_i
            p.line('x', 'y', source=source, color=color, **kw)
            label_set = True
        if '.' in style:
            kw = kwargs.copy()
            label_j = None if label_set else label_i
            if legend_loc != 'hide' and label_j is not None:
                kw['legend_label'] = label_j
            if hover:
                kw['name'] = label_i
            p.circle('x', 'y', source=source, color=color, **kw)

    if isinstance(hline, (int, float)):
        hline = [hline]
    if isinstance(hline, (list, tuple)):
        for y in hline:
            span = Span(location=y, dimension='width', line_color=hline_color, line_width=1, level='overlay')
            p.renderers.append(span)
    elif hline is not None:
        raise TypeError(f'Unsupported type of hline: {type(hline)}')

    if isinstance(vline, (int, float)):
        vline = [vline]
    if isinstance(vline, (list, tuple)):
        for x in vline:
            span = Span(location=x, dimension='height', line_color=vline_color, line_width=1, level='overlay')
            p.renderers.append(span)
    elif vline is not None:
        raise TypeError(f'Unsupported type of vline: {type(vline)}')
    if legend_loc != 'hide':
        if label is not None:
            p.legend.click_policy="hide"
        if legend_loc is not None:
            p.legend.location = legend_loc
    if xlabel is not None:
        p.xaxis.axis_label = xlabel
    if ylabel is not None:
        p.yaxis.axis_label = ylabel
    handle = None
    if notebook_handle:
        handle = bp.show(p, notebook_handle=notebook_handle)
        FIGURE.clear()
        return source, handle
    elif return_source:
        return source
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

def hist(x, nbins=30):
    hist, edges = np.histogram(x, density=True, bins=nbins)
    p = figure()
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="navy", line_color="white", alpha=0.5)
    bp.show(p)

def imshow(*ims, p=None, cmap='viridis', stretch=True, notebook_handle=False, show=True, 
           link=True, axes=False, toolbar=True, compact=True, merge_tools=True, toolbar_location='right'):
    if len(ims) > 1:
        ps = [imshow(im, cmap=cmap, stretch=stretch, axes=axes, toolbar=toolbar, compact=compact, show=False) 
                for i,im in enumerate(ims)]
        for pi in ps[1:]:
            pi.x_range = ps[0].x_range
            pi.y_range = ps[0].y_range
        return bp.show(bl.gridplot([ps], merge_tools=merge_tools, toolbar_location=toolbar_location))
    if isinstance(ims[0], (list, tuple)):
        ims = ims[0]
        if not isinstance(ims[0], (list, tuple)):
            ims = [ims]
        ps = []
        for i, ims_row in enumerate(ims):
            ps_row = [imshow(im, cmap=cmap, stretch=stretch, axes=axes, toolbar=toolbar, compact=compact, show=False) 
                      for i,im in enumerate(ims_row)]
            if link:
                if i == 0:
                    p0 = ps_row[0]
                    for pi in ps_row[1:]:
                        pi.x_range = p0.x_range
                        pi.y_range = p0.y_range
                else:
                    for pi in ps_row:
                        pi.x_range = p0.x_range
                        pi.y_range = p0.y_range
            ps.append(bl.gridplot([ps_row], merge_tools=merge_tools, toolbar_location=toolbar_location))
        return bp.show(bl.column(ps))

    im = ims[0]
    if p is None:
        width = 300 if compact else 400
        p = figure(int(width/im.shape[0]*im.shape[1]), width)   # width = 400, keep aspect ratio
    if axes is False:
        p.axis.visible=False
    if toolbar is False:
        p.toolbar.logo = None
        p.toolbar_location = None
    if np.issubdtype(im.dtype, np.floating):
        if stretch:
            _min, _max = im.min(), im.max()
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
    if len(im.shape) == 2:
        colormap = cm.get_cmap(cmap)
        palette = [matplotlib.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
        h = p.image([im], x=[0], y=[0], dw=[im.shape[1]], dh=[im.shape[0]], palette=palette)
    elif len(im.shape) == 3:
        if im.shape[-1] == 3: # rgb
            im = np.dstack([im, np.full_like(im[:,:,0], 255)])
        im1 = im.view(dtype=np.uint32).reshape(im.shape[:2])
        h = p.image_rgba(image=[np.flipud(im1)], x=[0], y=[0], dw=[im1.shape[1]], dh=[im1.shape[0]])
    else:
        raise ValueError('Unsupported image shape: ' + str(im.shape))
    if show:
        bp.show(p, notebook_handle=notebook_handle)
    else:
        return p
    if notebook_handle:
        return h

def show_df(df):
#    source = ColumnDataSource(df)
    source = ColumnDataSource({str(k): v for k, v in df.items()})
    columns = [
        TableColumn(field=str(q), title=str(q))
            for q in df.columns
    ] 
    data_table = DataTable(source=source, columns=columns, width=960)#, height=280)

    bp.show(data_table)

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
        xlabel=xlabel, ylabel=ylabel, xylabels=xylabels,
        RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK,
        push_notebook=push_notebook,
        bp=bp, bl=bl, imshow=imshow, hist=hist, show_df=show_df))

if __name__ == '__main__':
    #test_parse3()
    test_parse_arrs()
    test_parse_np()
    test_parse_pd()
    test_parse_dicts()
