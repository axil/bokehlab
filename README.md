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

Several plots: 
    - plot([1,2,3], [1,4,9], '.-')   # two plots, the second one is green
      plot([1,2,3], [1,8,27], '.-g') # in the same jupyter cell

    - y1 = [1,2,5]
      y2 = [2,3,10]
      plot([y1, y2])           # automatic colors       

    - x = [1,5,10]
      y1 = [1,2,5]
      y2 = [2,3,10]
      plot(x, [y1, y2])

    - y1 = [1,2,5]
      y2 = [2,3,10]
      plot([y1, y2], '.-bg')   # blue and green

    - y1 = [1,2,5]
      y2 = [2,3,10]
      plot([y1, y2], style=['.', '.-'], color=['b', 'g'])

    - plot([1,2,3], [1,4,9], '.-', [1,2,3], [1,8,27], '.-g')

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

Legend:
   - plot([1,2,3], [1,4,9], legend='plot1')
     plot([1,2,3], [2,5,10], legend='plot2')

   - plot([y1, y2], legend=['y1', 'y2'])

Legend location:
   plot([1,2,3], [1,4,9], legend='plot1', legend_loc='top_left')
   plot([1,2,3], [2,5,10], legend='plot2')

Other legend locations:
https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#location

## Advanced usage

`semilogx()`, `semilogy()` and `loglog()` show (semi)logarithmic plots with the same syntax as `plot()`.

`plot(x, y, hover=True)` displays point coordinates on mouse hover.
