import os
from IPython.core.magic import register_line_magic
from pathlib import Path

@register_line_magic
def bokehlab(line):
    """
    Magic equivalent to %load_ext bokehlab. Injects keywords like 'plot'
    into global namespace.
    """
    from bokehlab import CONFIG, load_config
    load_config()
    if line in ('cdn', 'inline', 'local'):
        CONFIG['resources'] = line
    elif line:
        print('unknown option')
    get_ipython().run_line_magic('load_ext', 'bokehlab')

@register_line_magic
def bokehlab_config(line):
    '''
    Configure bokehlab. Syntax: 
    %bokehlab_config [-g/--global] key=value [key1=value1 [...]]
      -g or --global saves config to ~/.bokeh/bokehlab.yaml
    For example, 
    %bokehlab_config width=500 height=200
    '''
    from bokehlab import CONFIG, CONFIG_DIR, CONFIG_FILE, load_config
    load_config()
    if not line:
        for k, v in CONFIG.items():
            print(f'{k} = {v}')
    else:
        parts = line.split()
        _global = False
        config = {}
        for part in parts:
            if part in ('-g', '--global'):
                print('global')
                _global = True
            elif '=' in part:
                k, v = part.split('=', 1)
                print(k, '=', v)
                if k in CONFIG.keys():
                    if k == 'resources':
                        CONFIG[k] = config[k] = v
                    else:
                        try:
                            CONFIG[k] = config[k] = int(v)
                        except:
                            print(f'{v} must be an integer')
                else:
                    print(f'Unknown key: {k} = {v}')
        if _global:
            import yaml
            if not CONFIG_DIR.exists():
                CONFIG_DIR.mkdir()
            if CONFIG_FILE.exists():
                on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
            else:
                on_disk = {}
            on_disk.update(config)
            CONFIG_FILE.open('w').write(yaml.dump(on_disk))
            print('config saved')
