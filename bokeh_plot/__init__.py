import bokeh.plotting as bp
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.io import output_notebook
from bokeh.layouts import layout
from bokeh.resources import INLINE
import numpy as np

__version__ = '0.1.5'

#output_notebook(resources=INLINE)
output_notebook()

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = '#ff7f0e'
RED = '#d62728'
COLORS = {'b': BLUE, 'g': GREEN, 'o': ORANGE, 'r': RED}

def figure(plot_width=950, plot_height=300, active_scroll='wheel_zoom', **kwargs):
    return bp.figure(plot_width=plot_width, plot_height=plot_height,
                     active_scroll=active_scroll, **kwargs)

def plot(*args, p=None, hover=False, mode='plot', **kwargs):
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
    if hover:
        p.add_tools(HoverTool(tooltips = [("x", "@x"),("y", "@y")]))
    tr = []
    style = '-'
    if len(args) in (1, 2):
        x = None
        if len(args) == 1:
            y = args[0]
        elif isinstance(args[1], str):
            y, style = args
        else:
            x, y = args
        if x is None:
            x = list(range(len(y)))
        tr.append((x, y, style))
    elif len(args) % 3 == 0:
        for h in range(len(args)//3):
            tr.append(args[3*h:3*(h+1)])
    base_color = kwargs.pop('color', BLUE)
    for x, y, style in tr:
        source = ColumnDataSource(data=dict(x=x, y=y))
        if style and style[-1] in COLORS:
            color = COLORS[style[-1]]
        else:
            color = base_color
        if style is None or '-' in style:
            p.line('x', 'y', source=source, color=color, **kwargs)
        if style is None or '.' in style:
            p.circle('x', 'y', source=source, color=color, **kwargs)
    if show:
        bp.show(p)

def semilogx(*args, **kwargs):
    kwargs['mode'] = 'semilogx'
    plot(*args, **kwargs)
    
def semilogy(*args, **kwargs):
    kwargs['mode'] = 'semilogy'
    plot(*args, **kwargs)
    
def loglog(*args, **kwargs):
    kwargs['mode'] = 'loglog'
    plot(*args, **kwargs)

def load_ipython_extension(ipython):
    ipython.user_ns.update(dict(figure=figure, plot=plot,
        semilogx=semilogx, semilogy=semilogy, loglog=loglog))
