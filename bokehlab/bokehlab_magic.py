import os
from collections import defaultdict

import yaml
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
    elif line in ('-v', '--verbose'):
        print('Using', CONFIG.get('resources', ''), 'resources')
    elif line:
        print(f'Unknown resources mode: "{line}". Available modes: {RESOURCE_MODES}')
    ip = get_ipython()
    if 'bokehlab' not in ip.extension_manager.loaded:
        ip.run_line_magic('load_ext', 'bokehlab')
    else:
        print('BokehJS already loaded, reloading...')
        ip.run_line_magic('reload_ext', 'bokehlab')

@register_line_magic
def bokehlab_config(line):
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
        if '--clear' in parts:
            if '--force' in parts or input('Are you sure you want to delete the configuration file (y/n)? ') == 'y':
                os.unlink(CONFIG_FILE)
                print('Config file deleted')
        elif '-d' in parts or '--delete' in parts:
            _global = False
            keys = []
            for part in parts:
                if part in ('-g', '--global'):
                    _global = True
                    print(f'Editing global settings in {CONFIG_FILE}')
                elif part in ('-d', '--delete'):
                    pass
                else:
                    if '.' in part:
                        section, key = part.split('.', 1)
                        if section in CONFIG:
                            if key in CONFIG[section]:
                                del CONFIG[section][key]
                                print(f'{part} deleted')
                            else:
                                print(f'{part} not found')
                            if not CONFIG[section]:
                                del CONFIG[section]
                    else:
                        if part in CONFIG:
                            del CONFIG[part]
                            print(f'{part} deleted')
                        else:
                            print(f'{part} not found')
                    keys.append(part)
            if _global:
                if not CONFIG_DIR.exists():
                    CONFIG_DIR.mkdir()
                if CONFIG_FILE.exists():
                    on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
                else:
                    on_disk = {}
                for part in keys:
                    if '.' in part:
                        section, key = part.split('.', 1)
                        if section in CONFIG:
                            on_disk[section].pop(key, None)
                            if not on_disk[section]:
                                del on_disk[section]
                    else:
                        on_disk.pop(part, None)
                CONFIG_FILE.open('w').write(yaml.dump(on_disk))
                print('Settings saved')
        else:
            _global = False
            config = defaultdict(dict)   # items to save to global config
            for part in parts:
                if part in ('-g', '--global'):
                    print(f'Editing global settings in {CONFIG_FILE}')
                    _global = True
                elif '=' in part:
                    k, v = part.split('=', 1)
                    if len(v)>1 and v[0] == v[-1] == "'":
                        v = v[1:-1]
                    elif v == 'None':
                        v = None
                    elif v == 'True':
                        v = True
                    elif v == 'False':
                        v = False
                    else:
                        try:
                            v = int(v)
                        except:
                            print(f'Type of "{v}" not recognized (use single or double quotes for strings)')
                            continue
                    if k in ('height', 'width'):
                        k = 'figure.' + k
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
            if _global:
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
                print('Settings saved')
