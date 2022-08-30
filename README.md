# BokehLab

BokehLab is an interactive plotting library with the familiar matplotlib/matlab syntax.  
Built upon the [Bokeh](https://bokeh.org/) visualization library. Works with both classic Jupyter notebooks and JupyterLab.

## Installation: 

    pip install bokehlab

To load this extension in jupyter notebook (both classic jupyter and jupyter lab):

    %load_ext bokehlab

Or even shorter:

    %bokehlab

To make the short syntax working, either run 

    python -m bokehlab.install_magic

Or manually copy `bokelab_magic.py` from the distribution directory to `~\.ipython\profile_default\startup`.

## Basic plotting:

    plot([1,4,9])             # dots 
    plot([1,4,9], '.-')       # line and dots 
    plot([1,2,3], [1,4,9])    # x and y 
    plot([1,2,3], [1,4,9], '.-')    # x, y and line style

## Several plots in one figure: 

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/simple.png" width="800">

## Interactive controls:

    click and drag = pan
    mouse wheel = zoom, 
    wheel on x axis = zoom horizontally
    wheel on y axis = zoom vertically

## Multiple plots syntax (equivalent ways to draw it):

    x = [1,5,10]
    y1 = [1,4,9]
    y2 = [1,8,27]

    - plot(x, y1, '.-')        # solid line with dots
      plot(x, y2, '.-g')       # the second plot is green

    - plot([y1, y2])           # auto x, auto colors       

    - plot(x, [y1, y2])

    - plot([y1, y2], '.-bg')   # blue and green

    - plot([y1, y2], style=['.', '.-'], color=['b', 'g'])

    - plot(x, y1, '.-', x, y2, '.-g')


The following markers are supported so far:

    '.' dots
    '-' line
    '.-' dots+line

The following colors are supported so far:

    'b' blue
    'g' green
    'r' red
    'O' orange  (capital O to avoid clashes with 'o' for open dots)
    
NB The color specifier must go after the marker if both are present.

## Legend:

    - plot([1,2,3], [1,4,9], label='plot1')
      plot([1,2,3], [2,5,10], label='plot2')

    - plot([y1, y2], label=['y1', 'y2'])

Legend location:

    - plot([1,2,3], [1,4,9], label='plot1', legend_loc='top_left')
      plot([1,2,3], [2,5,10], label='plot2')

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/legend.png" width="800">

Other legend locations:
https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#location

## Axes labels:
  
    - plot([1,2,3], xlabel='time', ylabel='value')
    - xlabel('time'); ylabel('value')
    - xylabels('time', 'value')

## Other uses:

* `semilogx()`, `semilogy()` and `loglog()` show (semi)logarithmic plots with the same syntax as `plot()`.

* `hist(x)` displays a histogram of x

* `plot(x, y, hover=True)` displays point coordinates on mouse hover.

* `plot(x, y, vline=1, hline=1.5, vline_color='red')` in addition to the (x, y) plot displays an infinite vertical line with x=1 and custom red color and an infinite horizontal line with y=1.5 and the default pink color.

## Visualizing Pandas Dataframes

* `plot(df)` plots all columns of the dataframe as separate lines on the same figure with column names 
displayed in the legend and with index taken as the x axis values. If the legend grows too long, it can 
be hidden with `legend_loc='hide'` (new in v0.1.13):
<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/pandas.png" width="800">

* `show_df(df)` displays pandas dataframe as a table (new in v0.1.14):
<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/datatable.png" width="800">

## Displaying Images

* `imshow(a)` displays an array as an image:

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/imshow.png" width="800">

Complete list of colormaps: [https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html](https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html)

* `imshow(im1, im2, ...)` shows several images side by side with linked panning and zooming (`link=False` to disable):

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/two_images.png" width="800">

* `imshow([[im1, im2, ...], [im3, im4, ... ], ...])` displays a matrix of images with panning and zooming linked row-wise:

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/imshow2x3.png" width="800">

See also a contour plot example in the bokeh gallery [page](https://docs.bokeh.org/en/latest/docs/gallery/image.html).

## Location of the JavaScript code

The Bokeh library consists of two parts: backend is written in Python, the frontend is in javascript. 

By default, Bokehlab (just like Bokeh) will get the required BokehJs code from the internet, from cdn.bokeh.org. This mode is called 'cdn' (=content delivery network). Generally it is fine, except that it doesn't work offline.

Another option is to bundle the javascript into the ipynb notebook:

    %bokehlab inline

It is also ok, except that the size of the ipynb file grows by ~6Mb. It would look reasonable if it made notebook work on a computer without Bokeh installed, but in reality the python part is also essential for the plots to work, so basically it is just a waste of disk space.
Bokehlab introduces a third option: 

    %bokehlab local

It serves javascript files from the locally installed Bokeh library. It both works offline and does not take any extra space. The only issue with this mode is that it needs a one-shot setup:

    pip install bokeh-resources
    python -m bokeh-resources.install

This mode can also be used in 'vanilla' Bokeh, see the instructions on github.

## Configuring the defaults

You can set the default size of the figure with %bokehlab_config magic command (or its shorter alias %blc): 

    %blc width=500 height=200

This size will apply to all figures in the current notebook. To make this change permanent, use -g (or --global flag):

    %blc -g width=500 height=200

It will save those values to ~/.bokeh/bokehlab.yaml and use them in the future Jupyter sessions.

You can also make Bokehlab remember your preferred mode of loading the javascript half of the library, so instead of always writing `%bokehlab local` in every ipynb file can do

    %blc -g resources=local

and `%bokehlab` will use locally served resources from now on.

Config is also capable or 'memorizing' the repeated arguments to any of the commands described above. For example, to tell Bokehlab to use thicker lines:
   
    %blc plot.line_width=2

and all subsequent calls to plot will assume line_width argument to be 2 (pixels) instead of one (this feature is work-in-progress, not all options are configurable yet).

To revert any configured option:

    %blc -d plot.line_width

A list of currently active settings is displayed with

    %blc

## Comparison to bokeh

Bokehlab is a thin wrapper over the excellent library `bokeh` primarily aimed at cutting down the amount of boilerplate code.

The following commands are equivalent:

<img src="https://raw.githubusercontent.com/axil/bokehlab/master/img/bokehlab_vs_bokeh.png" width="800">
