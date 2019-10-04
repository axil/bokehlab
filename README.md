# bokeh-plot

The following syntax is supported:

    plot([1,4,9])             # x is automatic 
    plot([1,4,9], '.-')       # line and dots 
    plot([1,2,3], [1,4,9])    # x and y 
    plot([1,2,3], [1,4,9], '.-')    # x, y and line style
    plot([1,2,3], [1,4,9], '.-', [1,2,3], [1,8,27], '.-g')   # two plots, the second one is green

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

There're also semilogx, semilogy and loglog for (semi)logarithmic plots.

## Installation

pip install bokeh-plot

## Usage

%load_ext bokeh_plot
