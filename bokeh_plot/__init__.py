import sys
import re
from collections.abc import Iterable
from collections import deque
from itertools import cycle
from datetime import datetime

USE_TORCH = 0

import bokeh.plotting as bp
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

__version__ = '0.1.18'

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

USE_TORCH = 0

if USE_TORCH:
    import torch
else:
    class torch:
        class Tensor:
            pass

def is_2d(y):
    return isinstance(y, torch.Tensor) and y.dim()==2 or \
       not isinstance(y, torch.Tensor) and isinstance(y, Iterable) and len(y) and \
       isinstance(y[0], Iterable) and not isinstance(y[0], str)

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

def parse3(x, y, spec): # -> list of (x, y, spec, label)
    tr = []
    #import ipdb; ipdb.set_trace()
    if is_2d(y):
        labels = None
        if isinstance(y, np.ndarray):
            if len(y.shape) != 2:
                raise ValueError(f'y is expected to be 1 or 2 dimensional, got {len(y.shape)} instead')
            yy = y.T
        elif isinstance(y, pd.DataFrame):
            labels = list(map(str, y.columns))
            yy = [y[col].values for col in y.columns]
        else:
            yy = y #list(col for col in zip(*y))
        n = len(yy)    # number of plots
        if labels is None:
            labels = [None] * n
#        if isinstance(x, Missing) and w == 2 and h > 2:
#            if isinstance(y, np.ndarray):
#                tr.append((y[:,0], y[:,1], spec))
#            else:
#                _x, _y = [], []
#                for row in y:
#                    _x.append(row[0])
#                    _y.append(row[1])
#                tr.append((_x, _y, spec))
#        else:
        #n = len(y)
        if isinstance(spec, (tuple, list)):
            specs = spec
            if len(specs) != w:
                raise ParseError(f'len(spec)={len(spec)} does not match len(y)={len(y)}')
        else:
            style, colors = parse_spec(spec)
            if len(colors) == n:
                specs = [style+c for c in colors]
            else:
                specs = [style+colors]*n
        if is_2d(x):
            if hasattr(x, 'T'):
                x = x.T
            for xi, yi, si in zip(x, yy, specs):
                tr.append((xi, yi, si))
        else:
            for yi, si, lb in zip(yy, specs, labels):
                if isinstance(x, Missing):
                    xi = np.arange(len(yi))
                else:
                    xi = x
                tr.append((xi, yi, si, lb))
    else:
        if isinstance(x, Missing):
            x = np.arange(len(y))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError(f'y is expected to be 1 or 2 dimensional, got {len(y.shape)} instead')
        if isinstance(y, pd.DataFrame) and len(y.columns) > 0:
            label = y.columns[0]
        else:
            label = None
        tr.append((x, y, ''.join(parse_spec(spec)), label))
    return tr

def compare(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for q, r in zip(a, b):
            if not compare(q, r):
                return False
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
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
        print(f'{repr(a)} != {repr(b)}')

def test_parse3():
    m = Missing()
    x = [1, 2, 3]
    y = [1, 4, 9]
    y1 = [-1, -4, -9]
    ax = [0, 1, 2]      # auto_x
    test(parse3(m, y, ''), [(ax, y, '-a', None)])
    test(parse3(x, y, ''), [(x, y, '-a', None)])
    test(parse3(m, [y, y1], ''), [(ax, y, '-a', None), (ax, y1, '-a', None)])
    test(parse3(x, [y, y1], ''), [(x, y, '-a', None), (x, y1, '-a', None)])
    test(parse3(x, [y, y1], ''), [(x, y, '-a', None), (x, y1, '-a', None)])
    test(parse3(x, [y, y1], '.'), [(x, y, '.a', None), (x, y1, '.a', None)])
    test(parse3(x, [y, y1], '.-'), [(x, y, '.-a', None), (x, y1, '.-a', None)])
    test(parse3(x, [y, y1], 'gr'), [(x, y, '-g', None), (x, y1, '-r', None)])
    test(parse3(x, [y, y1], '.-gr'), [(x, y, '.-g', None), (x, y1, '.-r', None)])
    print()
#    assert parse3([[1, 2], [3, 4], [5, 6]]) == [([0,1,2], [1,3,5], '-a'), ([0,1,2], [2,4,6], '-a')]

#def parse_args(*args, color=None, label=None):
#    return tr

def parse(*args, color=None, label=None):
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
    if isinstance(label, (list, tuple)):
        if len(label) != n:
            raise ValueError(f'len(label)={len(label)}; label should either be a string or a tuple of length {n}')
        labels = label
    elif isinstance(label, str):
        labels = [label] * n
    elif label is None:
        labels = [None] * n
    else:
        raise ValueError(f'label={label}; it should either be a string or a tuple of length {n}')
    qu = []
    try:
        for (x, y, spec, label1), color, label2 in zip(tr, colors, labels):
            style, _color = parse_spec(spec)
            if _color != 'a':
                color = _color
            qu.append((x, y, style, color, label2 if label2 is not None else label1))
    except Exception as e:
        print(e)
        
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
    test(parse(y), [([0, 1, 2], y, '-', 'a', None)])
    test(parse(x, y), [(x, y, '-', 'a', None)])
    test(parse(x, y, '.'), [(x, y, '.', 'a', None)])
    test(parse(x, y, '.-'), [(x, y, '.-', 'a', None)])
    test(parse(x, y, '.-g'), [(x, y, '.-', 'g', None)])
    test(parse(x, y, '.-g', label='aaa'), [(x, y, '.-', 'g', 'aaa')])
    test(parse(x, [y, y1], '.-', color=['r', 'g']), [(x, y, '.-', 'r', None), (x, y1, '.-', 'g', None)])
    test(parse(x, [y, y1], '.-rg', label=['y', 'y1']), [(x, y, '.-', 'r', 'y'), (x, y1, '.-', 'g', 'y1')])
    test(parse(x, [y, y1], '.-g', label='aaa'), \
       [([1, 2, 3], [1, 4, 9], '.-', 'g', 'aaa'),
        ([1, 2, 3], [-1, -4, -9], '.-', 'g', 'aaa')])
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


def plot(*args, p=None, hover=False, mode='plot', hline=None, vline=None, 
        color=None, hline_color='pink', vline_color='pink', 
        label=None, legend_loc=None, **kwargs):
#    print('(plot) FIGURE =', FIGURE)
    try:
        #show = p is None
        quintuples = parse(*args, color=color, label=label)
        is_dt = check_dt(quintuples)
        if p is None:
            if not FIGURE:
                kw = {}#'x_axis_type': None, 'y_axis_type': None}
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

        notebook_handle = kwargs.pop('notebook_handle', False)
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
            span = Span(location=hline, dimension='width', line_color=hline_color, line_width=1, level='overlay')
            p.renderers.append(span)
        if isinstance(vline, (int, float)):
            span = Span(location=vline, dimension='height', line_color=vline_color, line_width=1, level='overlay')
            p.renderers.append(span)
        if legend_loc != 'hide':
            if label is not None:
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

def hist(x, nbins=30):
    hist, edges = np.histogram(x, density=True, bins=nbins)
    p = figure()
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
            fill_color="navy", line_color="white", alpha=0.5)
    bp.show(p)

def imshow(im, p=None, cmap='viridis', stretch=True, notebook_handle=False):
    if p is None:
        p = figure(int(400/im.shape[0]*im.shape[1]), 400)   # height = 400, keep aspect ratio
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
    bp.show(p, notebook_handle=notebook_handle)
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
        RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK,
        push_notebook=push_notebook,
        bp=bp, imshow=imshow, hist=hist, show_df=show_df))

if __name__ == '__main__':
    test_parse3()
    test_parser()
