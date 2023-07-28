import math
from itertools import cycle
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from jupyter_bokeh import BokehModel
import ipywidgets as ipw
from IPython.display import display

#USE_TORCH = 0

import bokeh.plotting as bp
from bokeh.plotting.figure import Figure as BokehFigure
import bokeh.layouts as bl
import bokeh.models as bm
from bokeh.models import HoverTool, ColumnDataSource, Span, CustomJSHover, DataTable, TableColumn, \
    DatetimeAxis, Row, Column, GridBox
from bokeh.io import output_notebook, output_file, reset_output, push_notebook
from bokeh.resources import INLINE, CDN, Resources
from .config import CONFIG, CONFIG_LOADED, load_config, RESOURCE_MODES
from .install_magic import install_magic

#if USE_TORCH:
#    import torch
#else:
#    class torch:
#        class Tensor:
#            pass

import matplotlib       # for imshow palette
import matplotlib.cm as cm

__version__ = '0.2.10'

SIZING_METHOD = 'policies'        # or 'sizing_mode'

DEBUG_RESOURCES = False
DEBUG_OUTPUT = False

def load(resources=None):
    """
    Loads BokehJS 
    """
    load_config()
    if resources is None:
        resources = CONFIG.get('resources', {}).get('mode', 'cdn')
    res = None
    if resources == 'inline':
        res = INLINE
    elif resources == 'cdn':
        res = CDN
    elif resources == 'local':
        res = Resources('server', root_url='/nbextensions/bokeh_resources')
    elif resources == 'local-dev':
        res = Resources('server-dev', root_url='/nbextensions/bokeh_resources')
    if res:
        if DEBUG_RESOURCES:
            print(f'{resources} mode')
        mode = CONFIG.get('output', {}).get('mode', 'notebook')
        if DEBUG_OUTPUT:
            print(f'{mode} mode')
        if mode == 'notebook':
            output_notebook(res)
        elif mode == 'file':
            filename = CONFIG.get('output', {}).get('filename', None)
            if filename is not None:
                output_file(filename)
            else:
                reset_output()
    else:
        print(f'Unknown Bokeh resources mode: "{resources}", available modes: {RESOURCE_MODES}')

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
BLACK = '#000000'
COLORS = {'b': BLUE, 'g': GREEN, 'o': ORANGE, 'r': RED, 'k': BLACK}

#def get_color(c):
#    return "#1f77b4"
#    return 'b'
#    if c == 'a':
#        return next(AUTOCOLOR[0])
#    else:
#        return COLORS.get(c, c)
FIGURES = []
#AUTOCOLOR = []
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

bm.Model.model_class_reverse_map.pop('BokehlabFigure', None)       # to allow reload_ext

class Missing:
    pass

def expand_aliases(kw):
    if 'legend_loc' in kw:
        if 'legend_location' in kw and kw['legend_loc'] != kw['legend_location']:
            raise ValueError('Both legend_loc and legend_location are present and differ.')
        else:
            kw['legend_location'] = kw.pop('legend_loc')
    if 'toolbar_loc' in kw:
        if 'toolbar_location' in kw and kw['toolbar_loc'] != kw['toolbar_location']:
            raise ValueError('Both toolbar_loc and toolbar_location are present and differ.')
        else:
            kw['toolbar_location'] = kw.pop('toolbar_loc')

def process_max_size(kwargs, sizing_method=SIZING_METHOD):              # gridplot can only do 'sizing_mode'
    if sizing_method == 'policies':
        if kwargs.get('width') == 'max':
            del kwargs['width']
            kwargs['width_policy'] = 'max'
        elif kwargs.get('height') == 'max':
            del kwargs['height']
            kwargs['height_policy'] = 'max'
    elif sizing_method == 'sizing_mode':
        if kwargs.get('width') == 'max' and kwargs.get('height') == 'max':
            del kwargs['width']
            del kwargs['height']
            kwargs['sizing_mode'] = 'stretch_both'
        elif kwargs.get('width') == 'max':
            del kwargs['width']
            kwargs['sizing_mode'] = 'stretch_width'
        elif kwargs.get('height') == 'max':
            del kwargs['height']
            kwargs['sizing_mode'] = 'stretch_height'
    else:
        raise ValueError('Unknown sizing method. Must be either "policies" or "sizing_mode"')

class Figure(BokehFigure):
    __subtype__ = "BokehlabFigure"
    __view_model__ = "Plot"
    __view_module__ = "bokeh.models.plots"

    def __init__(self, width=None, height=None, *args, **kwargs):
        if width is not None:
            kwargs['width'] = width
        if height is not None:
            kwargs['height'] = height
        expand_aliases(kwargs)
        for k, v in CONFIG.get('figure', {}).items():
            if k not in kwargs:
                kwargs[k] = v
        self._autocolor = cycle(AUTOCOLOR_PALETTE)
        self._hover = kwargs.pop('hover', False)
        self._legend_location = kwargs.pop('legend_location', 
                                kwargs.pop('legend_loc', Missing))
        self._legend_title = kwargs.pop('legend_title', None)
        self._legend_added = False
        process_max_size(kwargs)
        if 'x_label' in kwargs:
            kwargs['x_axis_label'] = kwargs.pop('x_label')
        if 'y_label' in kwargs:
            kwargs['y_axis_label'] = kwargs.pop('y_label')
        if 'width' in kwargs:
            kwargs['plot_width'] = kwargs.pop('width')
        if 'height' in kwargs:
            kwargs['plot_height'] = kwargs.pop('height')
        grid = kwargs.pop('grid', True)
        flip_x_range = kwargs.pop('flip_x_range', False)
        flip_y_range = kwargs.pop('flip_y_range', False)
        super().__init__(*args, **kwargs)
        if not grid:
            self.xgrid.visible = False
            self.ygrid.visible = False
        if flip_x_range:
            self.x_range.flipped = True
        if flip_y_range:
            self.y_range.flipped = True

    def __enter__(self):
        FIGURES.append(self)
        return self

    def __exit__(self, type, value, traceback):
        FIGURES.pop()
    
    def _get_color(self, c):
        if c == 'a':
            return next(self._autocolor)
        else:
            return COLORS.get(c, c)

    def show(self, notebook_handle=False):
        return bp.show(self, notebook_handle=notebook_handle)

def figure(width=None, height=None, **kwargs):
    FIGURES.append(Figure(width=width, height=height, **kwargs))

#def Plot(*args, **kwargs):
#    kwargs['get_p'] = True
#    return plot(*args, **kwargs)

class Plot:
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        self.figure = plot(*args, **kwargs)
    
    def _ipython_display_(self):
        bp.show(self.figure)

    @property
    def x_range(self):
        return self.figure.x_range

    @x_range.setter
    def x_range(self, v):
        self.figure.x_range = v

    @property
    def y_range(self):
        return self.figure.y_range

    @y_range.setter
    def y_range(self, v):
        self.figure.y_range = v

class Stem(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        self.figure = stem(*args, **kwargs)    

class Semilogx(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        kwargs['mode'] = 'semilogx'
        self.figure = plot(*args, **kwargs)

class Semilogy(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        kwargs['mode'] = 'semilogy'
        self.figure = plot(*args, **kwargs)

class Loglog(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        kwargs['mode'] = 'loglog'
        self.figure = plot(*args, **kwargs)

class Imshow(Plot):
    def __init__(self, *args, **kwargs):
        kwargs['get_p'] = True
        self.figure = imshow(*args, **kwargs)

#def loglog_figure(width=None, height=None, 
#                  width_policy=None, height_policy=None, 
#                  active_scroll='wheel_zoom', **kwargs):
#    if width is None:
#        width = CONFIG['width']
#    if height is None:
#        height = CONFIG['height']
#    if width_policy is None:
#        width_policy = CONFIG['width_policy']
#    if height_policy is None:
#        height_policy = CONFIG['height_policy']
#    return bp.figure(plot_width=width, plot_height=height,
#                     width_policy=width_policy, height_policy=height_policy,
#                     active_scroll=active_scroll,
#                     x_axis_type='log', y_axis_type='log', **kwargs)

# ________________________________ parser __________________________________________

class ParseError(Exception):
    pass

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
    elif isinstance(v, (int, float, np.number)):
        v = [v] * n
    elif isinstance(v, (list, tuple)):
        if len(v) == 1:
            v *= n
        elif len(v) != n:
            raise ValueError(f'len({name})={len(v)} does not match len(y)={n}')
    return v
           
def parse(*args, x=None, y=None, style=None, color=None, label=None, source=None, default_style='-'):
    _x = _y = _style = _color = _label = None
    
    if len(args) == 1:
        _y = args[0]
    elif len(args) == 2:
        if is_string(args[1]):
            _y, _style = args
        else:
            _x, _y = args
    elif len(args) == 3:
        if is_string(args[1]):
            _y, _style, _color = args
        else:
            _x, _y, _style = args
    elif len(args) == 4:
        if is_string(args[1]):
            _y, _style, _color, _label = args
        else:
            _x, _y, _style, _color = args
    elif len(args) == 5:
        _x, _y, _style, _color, _label = args
    
    x = choose(x, _x, 'x')
    y = choose(y, _y, 'y')
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
            y = [y]
        else:
            if isinstance(y[0], (int, float, np.number)):      # flat list: [1,2,3]
                y = [y]
            elif isinstance(y[0], (list, tuple)):   
                if len(y[0]) > 0:
                    if isinstance(y[0][0], (int, float, np.number)):  # nested list: [[1,2,3],[4,5,6]]
                        pass
                    else:
                        raise TypeError(f'Unsupported y[0][0] type: {type(y[0][0])}')
            elif isinstance(y[0], np.ndarray):       
                if np.ndim(y[0]) == 0:               # list of numpy scalars [np.array(5), np.array(7)]
                    y = [y]
                else:                                # list of numpy arrays [np.array([1,2,3]), np.array([3,4,5])]      
                    pass
            elif isinstance(y[0], pd.Series):       
                if np.ndim(y[0]) == 0:               # list of pandas series [pd.Series([1,2,3]), pd.Series([4,5,6])]
                    y = [s.values for s in y]
                if x is None:
                    x = [s.index for s in y]
            elif isinstance(y[0], str):              # list of pandas column names
                if _y is not None:
                    raise TypeError('y of type str is only allowed as a keyword argument')
                if source is None:
                    raise TypeError('For y of type str a source must be provided')
                try:
                    for name in y:
                        if name not in source:
                            raise TypeError(f'{name} is missing from the source')
                    y = [source[name].values for name in y]
                    if x is None:
                        x = [source.index.values] * len(y)
                except:
                    raise 
            else:
                raise TypeError(f'Unsupported y[0] type: {type(y[0])}')

    elif isinstance(y, dict):
        if label is None:
            label = list(y.keys())
        y = list(y.values())
    
    elif isinstance(y, pd.Series):
        if x is None:
            x = [y.index]
        y = [y.values]
    
    elif isinstance(y, pd.DataFrame):
        if x is None:
            x = [y.index]
        if label is None:
            label = list(map(str, y.columns))
        y = [y[col].values for col in y.columns]

    elif isinstance(y, str):
        if _y is not None:
            raise TypeError('y of type str is only allowed as a keyword argument')
        if source is None:
            raise TypeError('For y of type str a source must be provided')
        if y in source:
            y = [source[y].values]
        else:
            raise TypeError(f'{y} is missing from the source')
    
    else:
        raise TypeError(f'Unsupported y type: {type(y)}')

    # By this point, y is a list of n arrays, each corresponding to a separate plot

    n = len(y)    # number of plots
    
    if x is None:
        x = [np.arange(len(yi)) for yi in y]

    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = [x] * n
        elif x.ndim == 2:
            if x.shape[1] > 1 and n == 1:
                n = x.shape[1]
                if isinstance(y, (list, tuple)):
                    y *= n
                elif isinstance(y, np.ndarray):
                    y = [y] * n
            elif x.shape[1] != n:
                raise ValueError(f'Wrong number of columns in x: expected {n}, got {x.shape[1]}')
            elif any(x.shape[0] != len(yi) for yi in y):
                raise ValueError(f'Wrong number of rows in x: got {x.shape[0]}')
            x = x.T
        else:
            raise ValueError(f'x is expected to be 1 or 2-dimensional, got {x.ndim} dimensions')

    elif isinstance(x, (list, tuple)):
        if len(x) == 0:
            if len(y) != 0:
                x = [x] * n
        elif isinstance(x[0], (int, float, np.number)):
            x = [x] * n
        elif isinstance(x[0], (list, tuple, np.ndarray, pd.Index)):
            if len(x) == 1:
                x *= n
            elif n == 1:
                n = len(x)
                if isinstance(y, (list, tuple)):
                    y *= n
                elif isinstance(y, np.ndarray):
                    y = [y] * n
            elif len(x) != n:
                raise ValueError(f'Number of x arrays = {len(x)} must either match number of y arrays = {n} or be equal to one')
        elif isinstance(x[0], str):
            if _x is not None:
                raise TypeError('x of type str is only allowed as a keyword argument')
            if source is None:
                raise TypeError('For x of type str a source must be provided')
            try:
                for name in x:
                    if name not in source:
                        raise TypeError(f'{name} is missing from the source')
                x = [source[name].values for name in x]
            except:
                raise 
        else:
            raise TypeError(f'Unsupported x[0] type: {type(x[0])}')

    elif isinstance(x, pd.Series):
        x = [x.values]*n

    elif isinstance(x, str):
        if _x is not None:
            raise TypeError('x of type str is only allowed as a keyword argument')
        if source is None:
            raise TypeError('For x of type str a source must be provided')
        if x in source:
            x = [source[x].values]
        else:
            raise TypeError(f'{x} is missing from the source')

    else:
        raise TypeError(f'Unsupported x type: {type(x)}')

    
    style = broadcast_str(style, n, default_style, 'style')
    color = broadcast_str(color, n, 'a', 'color')
    
    if label is None:
        label = [None] * n
    elif isinstance(label, str) and n == 1:
        label = [label]
    elif isinstance(label, (list, tuple)) and len(label) == n:
        if not isinstance(label[0], str):
            label = list(map(str, label))
    elif isinstance(label, np.ndarray) and len(label) == n:
        if not isinstance(label[0], str):
            label = list(map(str, label))
    elif isinstance(label, str):
        if n > 1:
            label = [label] * n
    else:
        raise ValueError(f'length of label = {len(label)} must match the number of plots = {n}')

    return list(zip(x, y, style, color, label))

# __________________________________________________________________________________


#class JupyterFigure(bp.Figure):
#    __subtype__ = "JupyterFigure"
#    __view_model__ = "Plot"
#
#    def _ipython_display_(self):
#        print('zzz')
#        bp.show(self)


#class JupyterFigure(bp.Figure):
#    __subtype__ = "JupyterFigure"
#    __view_model__ = "Plot"
#    __view_module__ = 'bokeh.models.plots'
#
#    def _ipython_display_(self):
#        bp.show(self)

class BokehWidget(BokehModel):

    def on_change(self, *args, **kwargs):
        self._model.on_change(*args, **kwargs)
        self.render_bundle = self._model_to_traits(self._model)
        self._model._update_event_callbacks()
    
    def on_event(self, *args, **kwargs):
        self._model.on_event(*args, **kwargs)
        self.render_bundle = self._model_to_traits(self._model)
        self._model._update_event_callbacks()

    def display(self):
        display(self)

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

def plot(*args, **kwargs):
    return _plot(*args, **kwargs)

def stem(*args, **kwargs):
    kwargs['stem'] = True
    return _plot(*args, **kwargs)

def Stem(*args, **kwargs):
    kwargs['get_p'] = True
    return stem(*args, **kwargs)

MARKER_STYLES = ['.', 'o', '*', 'x', '^', 'v', '<', '>']

ANGLES = {'<': math.pi/2, 'v': math.pi, '>': -math.pi/2}

LINE_STYLES = ['-', '--', ':', '-.']

def collect_figure_options(kw):
    res = {}
    for k in ('width', 'height', 'width_policy', 'height_policy', 'sizing_mode',
              'background_fill_color', 'x_range', 'y_range',
              'x_axis_location', 'y_axis_location', 
              'title', 'title_location', 'legend_location', 'legend_loc', 'grid',
              'toolbar_location', 'toolbar_loc', 'flip_x_range', 'flip_y_range',
              'legend_title'):
        if k in kw:
            res[k] = kw.pop(k)
    return res

def _plot(*args, x=None, y=None, style=None, color=None, label=None, line_width=None, alpha=None,
         p=None, hover=False, mode='plot', source=None,
         marker_size=None, fill_color=None, marker_line_width=None, 
         marker_color=None, line_color=None,
         hline=None, vline=None, hline_color='pink', vline_color='pink', 
         x_label=None, y_label=None, idx=None,
         #get_handle=False, 
         get_source=False, get_src=False, get_sh=False, get_ps=False, get_ws=False, show=True, get_p=False,
         flip_x_range=False, flip_y_range=False, stem=False, wrap=False, **kwargs):
    """
    The following parameters are passed to Figure if present:
        width, height, width_policy, height_policy, sizing_mode,
        background_fill_color, x_range, y_range,
        x_axis_location, y_axis_location, 
        title, title_location, legend_location, legend_loc, grid, 
        toolbar_location, 'flip_x_range', 'flip_y_range', 
        'legend_title'
    """
#    print('(plot) FIGURE =', FIGURE)
#    try:
    if len(args) == 0 and (x, y, hline, vline) == (None, None, None, None):
        args = [1], [1]
    if len(args) > 5:
        raise ValueError('Too many positional arguments, can not be more than 5')
    #show = p is None
    quintuples = parse(*args, x=x, y=y, style=style, color=color, label=label, source=source,
                       default_style='o-' if stem else '-')
    is_dt = check_dt(quintuples)
    figure_created = False
    toolbar_location_overridden = 'toolbar_location' in kwargs or 'toolbar_location' in CONFIG.get('figure', [])
    figure_opts = collect_figure_options(kwargs)
    if p is None:
        if not FIGURES:
            if mode == 'plot':
                pass
            elif mode == 'semilogx':
                figure_opts['x_axis_type'] = 'log'
            elif mode == 'semilogy':
                figure_opts['y_axis_type'] = 'log'
            elif mode == 'loglog':
                figure_opts['x_axis_type'] = figure_opts['y_axis_type'] = 'log'
            if is_dt:
                if 'x_axis_type' in figure_opts and figure_opts['x_axis_type'] is not None:
                    raise ValueError('datetime x values is incompatible with "%s"' % mode)
                else:
                    figure_opts['x_axis_type'] = 'datetime'
            expand_aliases(figure_opts)
            p = Figure(**figure_opts)
            if not (get_p or get_ps):
                if wrap:
                    FIGURES.append((p, 'wrap'))
                else:
                    FIGURES.append(p)
            figure_created = True
        else:
            p = FIGURES[-1]
            expand_aliases(figure_opts)
            if 'legend_location' in figure_opts:
                if p._legend_location is Missing:
                    p._legend_location = figure_opts['legend_location']
            if 'toolbar_location' in figure_opts:
                p.toolbar_location = figure_opts['toolbar_location']
    if not figure_created:
        if is_dt and not isinstance(p.xaxis[0], DatetimeAxis):
            raise ValueError('cannot plot datetime x values on a non-datetime x axis')
        elif not is_dt and isinstance(p.xaxis[0], DatetimeAxis):
            raise ValueError('cannot plot non-datetime x values on a datetime x axis')

    if hover is not None:
        p._hover = hover
    if p._hover:
        if is_dt:
            p.add_tools(HoverTool(tooltips=[('x', '@x{%F}'), ('y', '@y'), ('name', '$name')],
                        formatters={'@x': 'datetime'}))#, '@y': lat_custom}))
        else:
            p.add_tools(HoverTool(tooltips = [('idx', "@idx"), ("x", "@x"),("y", "@y"),('name', '$name')]))

    n = len(quintuples)

    line_widths = broadcast_num(line_width, n, None, 'line_width')
    alphas = broadcast_num(alpha, n, None, 'alpha')

    legend_not_empty = False
    sources = []
    hide_legend = p._legend_location is None
    for (x, y, style, color_str, label_i), line_width_i, alpha_i in zip(quintuples, line_widths, alphas):
        color = p._get_color(color_str) if hasattr(p, '_get_color') else '#1f77b4'
#        if isinstance(y, torch.Tensor):
#            y = y.detach().numpy()
        if len(x) != len(y):
            raise ValueError(f'len(x)={len(x)} is different from len(y)={len(y)}')
        if len(x) and isinstance(x[0], str):
            raise ValueError('plotting strings in x axis is not supported')
        if stem:
            data = dict(x=x, y0=np.zeros(len(x)), y=y)
        else:
            data = dict(x=x, y=y)
        if idx is not None:
            data['idx'] = idx
        source = ColumnDataSource(data=data)
        label_already_set = False
        if label_i:
            legend_not_empty = True
        if not style:
            if stem:
                style = 'o-'
            else:
                style = '-'
        if style[0] in MARKER_STYLES:
            marker_style = style[0]
            line_style = style[1:]
        else:
            marker_style= ''
            line_style = style
        if line_style:
            if line_style not in LINE_STYLES:
                raise ValueError(f'Unsupported plot style: {style}')
            kw = kwargs.copy()
            if not hide_legend and label_i is not None:
                kw['legend_label'] = label_i
            if hover:
                kw['name'] = label_i
            if line_width_i:
                kw['line_width'] = line_width_i
            if alpha_i:
                kw['alpha'] = alpha_i
            if line_style == ':':
                kw['line_dash'] = 'dotted'
            elif line_style == '-.':
                kw['line_dash'] = 'dotdash'
            elif line_style == '--':
                kw['line_dash'] = 'dashed'
            if stem:
                for k, v in CONFIG.get('segment', {}).items():
                    if k not in kw:
                        kw[k] = v
                p.segment('x', 'y0', 'x', 'y', source=source, color=color, **kw)
            else:
                for k, v in CONFIG.get('line', {}).items():
                    if k not in kw:
                        kw[k] = v
                p.line(x='x', y='y', source=source, color=color, **kw)
            label_already_set = True
        if marker_style:
            kw = kwargs.copy()
            label_j = None if label_already_set else label_i
            if not hide_legend and label_j is not None:
                kw['legend_label'] = label_j
            if hover:
                kw['name'] = label_i
            if marker_style != '.' and marker_size is None:
                marker_size = 7
            if marker_size:
                kw['size'] = marker_size
            if marker_style == 'o':
                if fill_color is None:
                    fill_color = 'white'
                if marker_line_width is None:
                    marker_line_width = 1.25
            if fill_color:
                kw['fill_color'] = fill_color
            if marker_line_width:
                kw['line_width'] = marker_line_width
            if marker_color:
                color = marker_color
            if color:
                kw['color'] = color
            if marker_style == '.':
                p.circle(x='x', y='y', source=source, **kw)
            elif marker_style == 'o':
                p.circle(x='x', y='y', source=source, **kw)
            elif marker_style == '*':
                p.asterisk(x='x', y='y', source=source, **kw)
            elif marker_style in '^v<>':
                if marker_style in ANGLES:
                    kw['angle'] = ANGLES[marker_style]
                p.triangle(x='x', y='y', source=source, **kw)
        sources.append(source)

    if isinstance(hline, (int, float, np.number)):
        hline = [hline]
    if isinstance(hline, (list, tuple)) or isinstance(hline, np.ndarray) and hline.ndim==1:
        for y in hline:
            span = Span(location=y, dimension='width', line_color=hline_color, 
                        line_width=1, level='overlay')
            p.renderers.append(span)
    elif hline is not None:
        raise TypeError(f'Unsupported type of hline: {type(hline)}')

    if isinstance(vline, (int, float, np.number)):
        vline = [vline]
    if isinstance(vline, (list, tuple)) or isinstance(vline, np.ndarray) and vline.ndim==1:
        for x in vline:
            span = Span(location=x, dimension='height', line_color=vline_color, 
                        line_width=1, level='overlay')
            p.renderers.append(span)
    elif vline is not None:
        raise TypeError(f'Unsupported type of vline: {type(vline)}')

    if legend_not_empty and not hide_legend:
        if label is not None:
            p.legend.click_policy="hide"
        if isinstance(p._legend_location, str) and not p._legend_added:
            loc = p._legend_location
            if loc in ('outside', 'top_outside', 'center_outside', 'bottom_outside'):
                loc = '_'.join(p._legend_location.rsplit('_', 1)[:-1]) # prefix
                p.add_layout(p.legend[0], 'right')
                if not toolbar_location_overridden and p.toolbar_location=='right':
                    p.toolbar_location = 'above'
            if loc:
                p.legend.location = loc
            p._legend_added = True
            # because p.legend is not ready until the first glyph is drawn
        if p._legend_title is not None:
            p.legend.title = p._legend_title
    if x_label is not None:
        p.xaxis.axis_label = x_label
    if y_label is not None:
        p.yaxis.axis_label = y_label
    handle = None
    if get_sh:
        handle = bp.show(p, notebook_handle=True)
#        FIGURE.clear()
        return sources[0] if len(sources)==1 else sources, handle
    elif get_ps:
#        FIGURE.clear()
        return p, sources[0] if len(sources)==1 else sources
    elif get_ws:
#        FIGURE.clear()
        return BokehWidget(p), sources[0] if len(sources)==1 else sources
#    elif show is False:
#        FIGURE.clear()
#        return p
    elif get_source or get_src:
        return sources[0] if len(sources)==1 else sources
    elif get_p:
        return p
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
    return plot(*args, **kwargs)

def semilogy(*args, **kwargs):
    kwargs['mode'] = 'semilogy'
    return plot(*args, **kwargs)

def loglog(*args, **kwargs):
    kwargs['mode'] = 'loglog'
    return plot(*args, **kwargs)

#def xlabel(label, p=None, **kw):
#    if p is None:
#        if not FIGURE:
#            p = figure(**kw)
#            FIGURE.append(p)
#        else:
#            p = FIGURE[0]
#    p.xaxis.axis_label = label
#
#def ylabel(label, p=None, **kw):
#    if p is None:
#        if not FIGURE:
#            p = figure(**kw)
#            FIGURE.append(p)
#        else:
#            p = FIGURE[0]
#    p.yaxis.axis_label = label
#
#def xylabels(xlabel, ylabel, p=None, **kw):
#    if p is None:
#        if not FIGURE:
#            p = figure(**kw)
#            FIGURE.append(p)
#        else:
#            p = FIGURE[0]
#    p.xaxis.axis_label = xlabel
#    p.yaxis.axis_label = ylabel

#def hist(x, nbins=30, width=None, height=None, get_p=False, get_ws=False, p=None, hover=False, **kwargs):
def hist(x, bins=30, range=None, hover=False, get_p=False, p=None, **kwargs):
    h = Hist(x, bins, range=range, hover=hover, get_p=get_p, **kwargs)
    if get_p:
        return h.figure

class Hist:
    def __init__(self, x, bins=30, range=None, get_p=True, get_ws=False, p=None, hover=False, mode=None, **kwargs):
        if p is None:
            if not FIGURES:
                kw = collect_figure_options(kwargs)
                if mode == 'semilogx':
                    kw['x_axis_type'] = 'log'
                elif mode == 'semilogy':
                    kw['y_axis_type'] = 'log'
                elif mode == 'loglog':
                    kw['x_axis_type'] = kw['y_axis_type'] = 'log'
                p = Figure(**kw)
                if not get_p:
                    FIGURES.append(p)
            else:
                p = FIGURES[-1] 
        values, edges = np.histogram(x, density=False, bins=bins, range=range)
        for k, v in CONFIG.get('segment', {}).items():
            if k not in kwargs:
                kwargs[k] = v
        defaults = dict(fill_color="navy", line_color="white", alpha=0.5)
        defaults.update(kwargs)
        if mode in ('semilogy', 'loglog'):
            bottom=np.ones_like(values)*0.5
        else:
            bottom=np.zeros_like(values)
        source = ColumnDataSource(data=dict(top=values, bottom=bottom, 
                                            left=edges[:-1], right=edges[1:]))
        p.quad(source=source, **defaults)
        self.histogram = values
        self.bin_edges = edges
        self.figure = p
        if hover:
            p.add_tools(HoverTool(tooltips = [("left", "@left"),("right", "@right"),("y", "@top")]))

    @property
    def widget(self):
        return BokehWidget(self.figure)
    
    def _ipython_display_(self):
        bp.show(self.figure)

def _ramp(palette, padding):
    return imshow(np.arange(256)[None, :].T.repeat(25, axis=1), palette=palette, width=35, show=False, 
                  toolbar=False, grid=False, padding=padding)

def old_calc_size(width, height, im_width, im_height, toolbar):
    if width == 'auto' and height == 'auto':    # by default, fix height, calculate widths
        height = CONFIG['imshow']['height']
        width = CONFIG['imshow']['width']
    if width == 'auto':        # calculate width from height, keeping aspect ratio
        if toolbar:
            width = int(height/im_height*im_width)+30
        else:
            width = int(height/im_height*im_width)
    if height == 'auto':       # calculate height from width, keeping aspect ratio
        if toolbar:
            height = int((width-30)/im_width*im_height)
        else:
            height = int(width/im_width*im_height)
    return width, height

#def calc_size(width, height, show_toolbar=False, show_colorbar=False):
#    if width == 'auto' and height == 'auto':    # by default, fix height, calculate widths
#        height = CONFIG['imshow']['height']
#        width = CONFIG['imshow']['width']
#    if width == 'auto':        # calculate width from height, keeping aspect ratio
#        if toolbar:
#            width = int(height/im_height*im_width)+30
#        else:
#            width = int(height/im_height*im_width)
#    if height == 'auto':       # calculate height from width, keeping aspect ratio
#        if toolbar:
#            height = int((width-30)/im_width*im_height)
#        else:
#            height = int(width/im_width*im_height)
#    return width, height

def mpl_cmap(name):
    colormap = cm.get_cmap(name)
    palette = [matplotlib.colors.rgb2hex(m) 
                for m in colormap(np.arange(colormap.N))]
    return palette
    

def imshow(*ims, p=None, palette='Viridis256', cmap=None, autolevels=True, show_axes=False, show_toolbar=True, 
           width=None, height=None, x_range=None, y_range=None,
           grid=True, flipud=False, hover=False, padding=0.1, 
           merge_tools=True, link=True, toolbar_location='right', show_colorbar=False, # multiple image related
           title=None, title_location=None, get_p=False,
           get_ws=False, notebook_handle=False, **kwargs):
    if len(ims) > 1:
#        if link:
#            max_height = max(im.shape[0] for im in ims)
#            max_width = max(im.shape[1] for im in ims)
#            width, height = calc_size(width, height, max_height, max_width, toolbar=False)
#            width, height = calc_size(width, height, max_height, max_width, show_toolbar=False)
        if title is not None:
            if len(title) != len(ims):
                raise ValueError(f'len(title) = {len(title)} must be the same as the number of images = {len(ims)}')
        else:
            title = [None] * len(ims)
        ps = [imshow(im, palette=palette, cmap=cmap, autolevels=autolevels, show_axes=show_axes, 
                     width=width, height=height, grid=grid, flipud=flipud, 
                     hover=hover, padding=padding, title=title[i], title_location=title_location,
                     show_toolbar=False if merge_tools else show_toolbar, 
                     show_colorbar=show_colorbar and i == len(ims)-1, get_p=True)
                for i, im in enumerate(ims)]
        if link:
            for pi in ps[1:]:
                pi.x_range = ps[0].x_range
                pi.y_range = ps[0].y_range
        if show_toolbar:
            grid = bl.gridplot([ps], merge_tools=merge_tools, toolbar_location=toolbar_location)
        else:
            grid = bl.gridplot([ps], merge_tools=False)
#        if show_colorbar:
#            grid = bl.row(grid, _ramp(palette=palette, padding=0))
        if get_p:
            return grid
        else:
            FIGURES.append(grid)
            return

#    if isinstance(ims[0], (list, tuple)):
#        ims = ims[0]
#        if not isinstance(ims[0], (list, tuple)):
#            ims = [ims]
#        ps = []
#        for i, ims_row in enumerate(ims):
#            ps_row = [imshow(im, palette=palette, autolevels=autolevels, show_axes=show_axes, show_toolbar=show_toolbar, 
#                      width=width, flipud=flipud, hover=hover, padding=padding) 
#                      for i,im in enumerate(ims_row)]
#            if link:
#                p0 = ps_row[0]
#                for pi in ps_row[1:]:
#                    pi.x_range = p0.x_range
#                    pi.y_range = p0.y_range
#            if show_colorbar:
#                ps.append(bl.row(
#                    bl.gridplot([ps_row], merge_tools=merge_tools, toolbar_location=toolbar_location),
#                    _ramp(palette=palette, padding=padding),
#                ))
#            else:
#                ps.append(bl.gridplot([ps_row], merge_tools=merge_tools, toolbar_location=toolbar_location))
#        return bp.show(bl.column(ps))

    im = ims[0]
    _padding = CONFIG.get('imshow', {}).get('padding', None)
    if _padding is not None:
        padding = _padding
    _show_axes = CONFIG.get('imshow', {}).get('show_axes', None)
    if _show_axes is not None:
        show_axes = _show_axes
    if p is None:
        kw = {}
        if hover is True:
            kw['tooltips'] = [("x", "$x"), ("y", "$y"), ("value", "@image")]
        if title is not None:
            kw['title'] = title
        if title_location is not None:
            kw['title_location'] = title_location
#        if not flipud:                 this does not work due to an issue in bokeh
#            p.y_range.flipped=True     workaround below 
        y_pad = im.shape[1]*padding/2 if padding is not None else 0
        if y_range is not None:
            kw['y_range'] = y_range
        else:
            if not flipud:
                kw['y_range'] = [im.shape[0]+y_pad, -y_pad]
            else:
                kw['y_range'] = [-y_pad, im.shape[0]+y_pad]
        x_pad = im.shape[0]*padding/2 if padding is not None else 0  # just for symmetry with y
        if x_range is not None:
            kw['x_range'] = x_range
        else:
            kw['x_range'] = [-x_pad, im.shape[1]+x_pad]

        if width is not None:
            kw['width'] = width
        else:
            kw['width'] = CONFIG.get('imshow', {}).get('width', None) or 300
        if height is not None:
            kw['height'] = height
        else:
            kw['height'] = CONFIG.get('imshow', {}).get('height', None) or 300

        if show_colorbar:
            kw['width'] += 70
        if show_toolbar:
            kw['width'] += 30
#        kw['width'], kw['height'] = calc_size(kw['width'], kw['height'], im.shape[1], im.shape[0], toolbar)
        p = Figure(**kw)
        if title_location is not None:
            p.title.align = 'center'
        if not get_p:
            FIGURES.append(p)

#    if padding is not None:            can be uncommented once the issue is resolved
#        # p.x_range.range_padding = p.y_range.range_padding = padding
#        p.x_range.range_padding = padding
    
    if grid is False:
        p.xgrid.visible = False
        p.ygrid.visible = False
    
    if show_axes is False:
        p.axis.visible=False
    if show_toolbar is False:
        p.toolbar.logo = None
        p.toolbar_location = None
    im = im.squeeze()
    _min, _max = 0, 255
    if np.issubdtype(im.dtype, np.floating):
        if autolevels:
            _min, _max = im.min(), im.max()
        else:
            _min, _max = 0., 1.
#            if _min == _max:
#                if _min > 1.:
#                    im = np.ones_like(im, dtype=np.uint8)
#                elif _min < 0.:
#                    im = np.zeros_like(im, dtype=np.uint8)
#            else:
#                im = (im-_min)/(_max-_min)
#        im = (im*255).astype(np.uint8)
    elif im.dtype == np.uint8:
        pass
    elif np.issubdtype(im.dtype, np.integer):
        if autolevels:
            _min, _max = im.min(), im.max()
#            if _min == _max:
#                im = np.zeros_like(im, dtype=np.uint8)
#            else:
#                im = ((im-_min)/(_max-_min)*255).astype(np.uint8)
    elif np.issubdtype(im.dtype, bool):
        _min, _max = 0, 1
    if im.ndim in (2, 3):
        if im.ndim == 2:
#            colormap = cm.get_cmap(palette)
#            palette = [matplotlib.colors.rgb2hex(m) 
#                       for m in colormap(np.arange(colormap.N))]
            if palette is None:            
                palette = 'Greys256'
            color_mapper = bm.LinearColorMapper(palette=palette, low=_min, high=_max)
        else:
            if im.shape[-1] == 3: # 3 is rgb; 4 means rgba already
                im = np.dstack([im, np.full_like(im[:,:,0], 255)])
            elif im.shape[-1] != 4:
                raise ValueError(f'Image array must be either (..., 3) or (..., 4), got {im.shape} instead')
            im = im.view(dtype=np.uint32).reshape(im.shape[:2])
#            palette = None
            color_mapper = None
        if not flipud:
            im = np.flipud(im)
        kw = dict(image=[im], x=[0], dw=[im.shape[1]])
        if not flipud:
            kw.update(dict(y=[im.shape[0]], dh=[im.shape[0]]))
        else:
            kw.update(dict(y=[0], dh=[im.shape[0]]))
        source = ColumnDataSource(data=kw)
        if color_mapper is not None:
            h = p.image(source=source, color_mapper=color_mapper)
        else:
            h = p.image_rgba(source=source)
    else:
        raise ValueError('Unsupported image shape: ' + str(im.shape))

    if get_ws:
        return BokehWidget(p), source
    if show_colorbar:
        color_bar = bm.ColorBar(color_mapper=color_mapper, label_standoff=12)
        p.add_layout(color_bar, 'right')
#        p = bl.row(p, _ramp(palette=palette, padding=0))
#    else:
#        bp.show(p, notebook_handle=notebook_handle)
    if get_p:
        return p
#    if notebook_handle:
#        return h

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

#class JupyterColumn(bm.Column):
#    __subtype__ = "JupyterColumn"
#    __view_model__ = "Column"
#    __view_module__ = 'bokeh.models.layouts'
#
#    def _ipython_display_(self):
#        bp.show(self)

def hstack(*args, merge_tools=False, toolbar_location='right', wrap=False, active_drag=None, link_x=False, link_y=False, **kwargs):
    args = [a.figure if isinstance(a, (Plot, Hist)) else a for a in args]
    figures = [arg for arg in args if isinstance(arg, bl.LayoutDOM)]
    if len(figures) > 1 and link_x:
        for fig in figures[1:]:
            fig.x_range = figures[0].x_range
    if len(figures) > 1 and link_y:
        for fig in figures[1:]:
            fig.y_range = figures[0].y_range
    all_bokeh = all(isinstance(arg, bl.LayoutDOM) for arg in args)
    if 'width' not in kwargs:
        if CONFIG.get('figure', {}).get('width', None) == 'max':
            kwargs['width'] = 'max'
    if 'height' not in kwargs:
        if CONFIG.get('figure', {}).get('height', None) == 'max':
            kwargs['height'] = 'max'
    process_max_size(kwargs, 'sizing_mode')
    if all_bokeh:
#        for k in ('width_policy', 'height_policy'):
#            v = CONFIG['figure'].get(k)
#            if v not in ('auto', 'fixed', None):
#                kwargs[k] = v
        if merge_tools:
            p = bl.gridplot([args], merge_tools=True, 
                            toolbar_location=toolbar_location, **kwargs)
            if active_drag is not None:
                args[0].toolbar.active_drag = active_drag
        else:
            p = bl.row(*args, **kwargs)
        if wrap:
            return BokehWidget(p)
        else:
            return p
    else:
        if merge_tools:
            raise ValueError('Can only merge tools if all arguments are Bokeh objects (not widgets)')
        converted = [BokehWidget(arg) if isinstance(arg, bl.LayoutDOM) else arg for arg in args]
        return ipw.HBox(converted)

def vstack(*args, merge_tools=False, toolbar_location='right', wrap=False, active_drag=None, link_x=False, link_y=False, **kwargs):
    args = [a.figure if isinstance(a, (Plot, Hist)) else a for a in args]
    figures = [arg for arg in args if isinstance(arg, bl.LayoutDOM)]
    if len(figures) > 1 and link_x:
        for fig in figures[1:]:
            fig.x_range = figures[0].x_range
    if len(figures) > 1 and link_y:
        for fig in figures[1:]:
            fig.y_range = figures[0].y_range
    all_bokeh = all(isinstance(arg, bl.LayoutDOM) for arg in args)
    if 'width' not in kwargs:
        if CONFIG.get('figure', {}).get('width', None) == 'max':
            kwargs['width'] = 'max'
    if 'height' not in kwargs:
        if CONFIG.get('figure', {}).get('height', None) == 'max':
            kwargs['height'] = 'max'
    process_max_size(kwargs, 'sizing_mode')

    if all_bokeh:
#        for k in ('width_policy', 'height_policy'):
#            v = CONFIG['figure'].get(k)
#            if v == 'max':
#                kwargs[k] = v
        if merge_tools:
#            kw = {}
#            for k in 'width', 'height', 'width_policy', 'height_policy':
#                kw[k] = kwargs[k]
            p = bl.gridplot([[a] for a in args], merge_tools=True, 
                            toolbar_location=toolbar_location, **kwargs)
#            for kw in 'active_drag', 'active_scroll', 'active_tap', 'active_inspect':
#                if kw in kwargs:
#                    setattr(args[0].toolbar, kw, kwargs[kw])
            if active_drag is not None:
                args[0].toolbar.active_drag = active_drag
        else:
            p = bl.column(*args, **kwargs)
        if wrap:
            return BokehWidget(p)
        else:
            return p
    else:
        if merge_tools:
            raise ValueError('Can only merge tools if all arguments are Bokeh objects (not widgets)')
        converted = [BokehWidget(arg) if isinstance(arg, bl.LayoutDOM) else arg for arg in args]
        return ipw.VBox(converted)

class AutoShow(object):
    def __init__(self):
#        self.shell = ip
#        self.last_x = None
        self.figures = []

#    def pre_execute(self):
#        self.last_x = self.shell.user_ns.get('x', None)
#
    def pre_run_cell(self, info):
        FIGURES.clear()
#        AUTOCOLOR.clear()
#        AUTOCOLOR.append(cycle(AUTOCOLOR_PALETTE))
    pre_run_cell.bokeh_plot_method = True
#        print('pre_run Cell code: "%s"' % info.raw_cell)

#    def post_execute(self):
#        if self.shell.user_ns.get('x', None) != self.last_x:
#            print("x changed!")
#
    def post_run_cell(self, result):
#        print('post run')
#        print('Cell code: "%s"' % result.info.raw_cell)
        if result.error_before_exec:
            print('Error before execution: %s' % result.error_before_exec)
        else:
#            p = self.shell.user_ns.get('FIGURE', [])
            for p in FIGURES:
                if isinstance(p, tuple):
                    display(BokehWidget(p[0]))
                else:
                    bp.show(p)
            FIGURES.clear()
    post_run_cell.bokeh_plot_method = True

def register_callbacks(ip):
    # Avoid re-registering when reloading the extension
    def register(event, function):
        for f in ip.events.callbacks[event]:
            if hasattr(f, 'bokeh_plot_method'):
                ip.events.unregister(event, f)
#                print('unregistered')
        ip.events.register(event, function)

    autoshow = AutoShow()
#    ip.events.register('pre_execute', vw.pre_execute)
    register('pre_run_cell', autoshow.pre_run_cell)
#    ip.events.register('post_execute', vw.post_execute)
    register('post_run_cell', autoshow.post_run_cell)

def load_ipython_extension(ip):
    load()
    register_callbacks(ip)
    mode = CONFIG.get('globals', {}).get('mode', 'normal')
    if mode == 'none':
        return
    d = dict(
        Figure=Figure, figure=figure,
        plot=plot, Plot=Plot, stem=stem, Stem=Stem, hist=hist, Hist=Hist,
        semilogx=semilogx, semilogy=semilogy, loglog=loglog,
        Semilogx=Semilogx, Semilogy=Semilogy, Loglog=Loglog,
        push_notebook=push_notebook, BokehWidget=BokehWidget,
        imshow=imshow, Imshow=Imshow, show_df=show_df,
        hstack=hstack, vstack=vstack)
    if mode == 'all':
        d.update(dict(
            bp=bp, bl=bl, bm=bm,
            RED=RED, GREEN=GREEN, BLUE=BLUE, ORANGE=ORANGE, BLACK=BLACK,
        ))
    ip.user_ns.update(d)

def gen_plot_wrapper(method):
    class _plot_wrapper:
        def __init__(self, _figure):
            self._figure = _figure
            
        def __call__(self, *args, **kwargs):
            kwargs.update(p=self._figure)
            return method(*args, **kwargs)

        def __getattr__(self, attr):
            return getattr(self._figure, attr)
    return _plot_wrapper

def _stem_wrapper(*args, **kwargs):
    kwargs['p'] = args[0]
    args = args[1:]
    return stem(*args, **kwargs)

def _imshow_wrapper(*args, **kwargs):
    kwargs['p'] = args[0]
    args = args[1:]
    return imshow(*args, **kwargs)

def _xlabel(self, label):
    self.xaxis.axis_label = label
    return self

def _ylabel(self, label):
    self.yaxis.axis_label = label
    return self

def _xylabel(self, x_label, y_label):
    self.xaxis.axis_label = x_label
    self.yaxis.axis_label = y_label
    return self

# Monkey-patching

#Figure._plot = Figure.plot
BokehFigure.plot = property(gen_plot_wrapper(plot))
BokehFigure.semilogx = property(gen_plot_wrapper(semilogx))
BokehFigure.semilogy = property(gen_plot_wrapper(semilogy))
BokehFigure.loglog = property(gen_plot_wrapper(loglog))
BokehFigure.stem = _stem_wrapper
BokehFigure.imshow = _imshow_wrapper
BokehFigure.xlabel = _xlabel
BokehFigure.ylabel = _ylabel
BokehFigure.xylabel = _xylabel

BokehFigure._ipython_display_ = lambda self: bp.show(self)
Row._ipython_display_ = lambda self: bp.show(self)
Column._ipython_display_ = lambda self: bp.show(self)
GridBox._ipython_display_ = lambda self: bp.show(self)

def _show(self, notebook_handle=False):
    bp.show(self, notebook_handle=notebook_handle)

BokehFigure.show = _show
Row.show = _show
Column.show = _show


#elif __name__ == 'bokehlab':
#    import IPython
#    ip = IPython.core.interactiveshell.InteractiveShell.instance()
#    register_callbacks(ip)
