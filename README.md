# bokeh-plot

## Installation: 

    pip install bokeh-plot

## Usage:

To load this extension in jupyter notebook:

    %load_ext bokeh_plot

The following syntax is supported:

    plot([1,4,9])             # x is automatic 
    plot([1,4,9], '.-')       # line and dots 
    plot([1,2,3], [1,4,9])    # x and y 
    plot([1,2,3], [1,4,9], '.-')    # x, y and line style
    plot([1,2,3], [1,4,9], '.-', [1,2,3], [1,8,27], '.-g')   # two plots, the second one is green

<img src="https://raw.githubusercontent.com/axil/bokeh-plot/master/img/screenshot.png" width="800">

The following markers are supported so far:

    '.' dots
    '-' line
    '.-' dots+line

The following colors are supported so far:

    'b' blue
    'g' green
    'r' red
    'o' orange
    
NB The color specifier must go after the marker if both are present.


## Advanced usage

`semilogx()`, `semilogy()` and `loglog()` show (semi)logarithmic plots with the same syntax as `plot()`.

`plot(x, y, hover=True)` displays point coordinates on mouse hover.
