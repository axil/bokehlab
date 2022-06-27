from collections import defaultdict

from IPython.core.magic import register_line_magic

@register_line_magic
def bokehlab(line):
    """
    Magic equivalent to %load_ext bokehlab. Injects keywords like 'plot'
    into global namespace.
    """
    from bokehlab import CONFIG, load_config, RESOURCE_MODES
    load_config()
    if line in RESOURCE_MODES:
        CONFIG['resources'] = line
    elif line:
        print(f'Unknown resources mode: "{line}". Available modes: {RESOURCE_MODES}')
    get_ipython().run_line_magic('load_ext', 'bokehlab')

@register_line_magic
def bokehlab_config(line):
    '''
    Configure bokehlab. Syntax: 
    
    %bokehlab_config [-g/--global] key=value [key1=value1 [...]]
      -g or --global saves config to ~/.bokeh/bokehlab.yaml
    For example, 
    %bokehlab_config width=500 height=200
    
    %bokehlab_config without arguments displays current config
    '''
    from bokehlab import CONFIG, CONFIG_DIR, CONFIG_FILE, load_config, RESOURCE_MODES
    load_config()
    if not line:
        for k, v in CONFIG.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f'{k}.{kk}={vv!r}')
            else:
                print(f'{k}={v!r}')
    else:
        parts = line.split()
        _global = False
        config = defaultdict(dict)   # items to save to global config
        for part in parts:
            if part in ('-g', '--global'):
                print('global')
                _global = True
            elif '=' in part:
                k, v = part.split('=', 1)
                if len(v)>1 and v[0] == v[-1] == "'":
                    v = v[1:-1]
                elif v == 'None':
                    v = None
                else:
                    try:
                        v = int(v)
                    except:
                        print(f'{v} must be an integer')
                        continue
                print(k, '=', repr(v))
                if k == 'resources':
                    if v in RESOURCE_MODES:
                        CONFIG[k] = config[k] = v
                    else:
                        print(f'Unknown resources mode: "{v}". Available modes: {RESOURCE_MODES}')
                elif '.' in k:
                    section, key = k.split('.', 1)
                    if section in ('figure', 'imshow'):
                        CONFIG[section][key] = config[section][key] = v
                    else:
                        print(f'Unknown section: {section}')
                else:
                    print(f'Unknown key: {k}')
        if 'width' in config['figure'] and config['figure'].get('width_policy') not in ('fixed', 'auto'):
            CONFIG['figure']['width_policy'] = config['figure']['width_policy'] = 'auto'
            print('figure.width implies figure.width_policy="auto"')
        if 'height' in config['figure'] and config['figure'].get('height_policy') not in ('fixed', 'auto'):
            CONFIG['figure']['width_policy'] = config['figure']['height_policy'] = 'auto'
            print('figure.height implies figure.height_policy="auto"')
        if _global:
            import yaml
            if not CONFIG_DIR.exists():
                CONFIG_DIR.mkdir()
            if CONFIG_FILE.exists():
                on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
            else:
                on_disk = {}
            for k, v in config.items():
                if k in on_disk and isinstance(v, dict):
                    for kk, vv in v.items():
                        on_disk[k][kk] = vv
                else:
                    on_disk[k] = v 
            CONFIG_FILE.open('w').write(yaml.dump(on_disk))
            print('config saved')
