from IPython.core.display import display, HTML
from IPython.core.magic import register_line_magic, register_line_cell_magic

@register_line_magic
def bokehlab(line):
    """
    Magic equivalent to %load_ext bokehlab. Injects keywords like 'plot'
    into global namespace.
    """
    from bokehlab import CONFIG, load_config, RESOURCE_MODES
    load_config()
    parts = line.split()
    verbose = False
    if '-v' in parts:
        parts.remove('-v')
        verbose = True
    if '--verbose' in parts:
        parts.remove('-v')
        verbose = True
    line = ' '.join(parts)
    if line in RESOURCE_MODES:
        CONFIG['resources'] = {'mode': line}
    elif line:
        print(f'Unknown resources mode: "{line}". Available modes: {RESOURCE_MODES}')
    if verbose:
        print('Using', CONFIG.get('resources', {}).get('mode'), 'resources')
    ip = get_ipython()
    if 'bokehlab' not in ip.extension_manager.loaded:
        ip.run_line_magic('load_ext', 'bokehlab')
    else:
        display(HTML('<div class="bk-root">BokehJS already loaded, reloading...</div>'))
        ip.run_line_magic('reload_ext', 'bokehlab')

@register_line_cell_magic
def bokehlab_config(line, cell=None):
    '''
    Configure bokehlab. Syntax: 
    
    1) %bokehlab_config [-g/--global] key=value [key1=value1 [...]]
      -g or --global saves config to ~/.bokeh/bokehlab.yaml
    For example, 
    %bokehlab_config figure.width=500 figure.height=200

    2) %bokehlab_config [-g/--global] -d/--delete key [key1 [...]]
    deletes the corresponding keys

    3) %bokehlab_config without arguments displays current config

    4) %bokehlab --clear deletes ~/.bokeh/bokehlab.yaml
    '''
    from bokehlab.config import configure
    configure(line, cell)

@register_line_cell_magic
def blc(line, cell=None):
    return bokehlab_config(line, cell)


