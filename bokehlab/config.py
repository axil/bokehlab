import os
from collections import defaultdict
from copy import deepcopy
import textwrap
from pathlib import Path

import yaml
from IPython.core.display import display, HTML

CONFIG = {
    'figure': {
        'width': 'max',
        'height': 300,
        'active_scroll': 'wheel_zoom',
    },
#    'imshow': {
#        'aspect_ratio': 1,
#    },
    'resources': { 
        'mode': 'cdn',
    },
    'output': {
        'mode': 'notebook',
    },
}
CONFIG_DIR = Path('~/.bokeh').expanduser()
CONFIG_FILE = CONFIG_DIR / 'bokehlab.yaml'
CONFIG_LOADED = False
CONFIG_SECTIONS = 'figure', 'imshow', 'resources', 'output', 'line', 'circle', 'globals'
#FIGURE_OPTIONS = set(CONFIG) - set('resources')  # all config keys except 'resources'
DEBUG_CONFIG = False
RESOURCE_MODES = ['cdn', 'inline', 'local', 'local-dev']

def load_config():
    global CONFIG_LOADED
    if not CONFIG_LOADED:
        if CONFIG_FILE.exists():
            on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
            for k, v in on_disk.items():
                if isinstance(v, dict) and k in CONFIG:
                    for kk, vv in v.items():
#                        if kk not in CONFIG[k]:
                        CONFIG[k][kk] = vv
                else:
                    CONFIG[k] = v
            CONFIG_LOADED = True
            if DEBUG_CONFIG:
                print(f'config loaded: {on_disk}')
    elif DEBUG_CONFIG:
        print('config already loaded')

def parse_config_line(parts, CONFIG, config=None, verbose=True):
    if '-v' in parts or '--verbose' in parts:
        verbose = True
    for part in parts:
        if '=' not in part:
            if '.' in part:
                section, key = part.split('.', 1)
                print(f'{part}=' + CONFIG.get(section, {}).get(key, 'missing'))
            continue
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
        if k == 'resources':
            k = 'resources.mode' + k
        if verbose:
            print(k, '=', repr(v))
        if '.' in k:
            if k == 'resources.mode' and v not in RESOURCE_MODES:
                print(f'Unknown resources mode: "{v}". Available modes: {RESOURCE_MODES}')
            else:
                section, key = k.split('.', 1)
                if section in CONFIG_SECTIONS:
                    if section not in CONFIG: 
                        CONFIG[section] = {key: v}
                    else:
                        CONFIG[section][key] = v
                    if config is not None:
                        config[section][key] = v
                else:
                    print(f'Unknown section: {section}. Available sections: {CONFIG_SECTIONS}')
        else:
            print(f'Unknown key: {k}')

def read_config():
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir()
    if CONFIG_FILE.exists():
        on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
    else:
        on_disk = {}
    return on_disk

def configure(line, cell=None):
    '''
    Configures bokehlab. Syntax:
    
    1) %bokehlab_config [-g/--global] key=value [key1=value1 [...]]
    without -g or --global configures currently active notebook
    with -g or --global saves config to ~/.bokeh/bokehlab.yaml for future sessions

    For example, 
    %bokehlab_config figure.width=500 figure.height=200

    2) %bokehlab_config [-g/--global] -d/--delete key [key1 [...]]
    deletes the corresponding keys locally (default) or globally (if -g or --global is present)

    3) %bokehlab_config 
    (without arguments) displays current config

    4) %bokehlab --clear 
    deletes ~/.bokeh/bokehlab.yaml

    5) %bokehlab -h/--help [key]
    displays help for the specific key or this message if the key is not given
    '''
    load_config()
    if cell is not None:
#        print('got', line, cell)
        config_backup = deepcopy(CONFIG)
        parse_config_line(line.split(), CONFIG, verbose=False)
        get_ipython().run_cell(cell)      ##
#        print('done')
        CONFIG.clear()
        CONFIG.update(config_backup)
    elif not line:
        for k, v in CONFIG.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    print(f'{k}.{kk}={vv!r}')
            else:
                print(f'{k}={v!r}')
    else:
        parts = line.split()
        if '-h' in parts or '--help' in parts:
            for part in parts:
                if part in ('-h', '--help'):
                    pass
                elif part == 'output.mode':
                    print("Available output modes: 'notebook', 'file'. For 'file' mode you can specify filename by setting output.file")
                    break
                elif part == 'resources.mode':
                    print('Available resource modes: ' + ', '.join(f'"{m}"' for m in RESOURCE_MODES))
                    break
            else:
                print(textwrap.dedent(configure.__doc__.strip('\n')))
        if '-g' in parts or '--global' in parts:
            _global = True
            print(f'Editing global settings in {CONFIG_FILE}')
        else:
            _global = False
        if '--clear' in parts:
            if '--force' in parts or input('Are you sure you want to delete the configuration file (y/n)? ') == 'y':
                os.unlink(CONFIG_FILE)
                print('Config file deleted')
        elif '-d' in parts or '--delete' in parts:
            _global = False
            keys = []
            for part in parts:
                if part in ('-g', '--global'):
                    pass
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
                on_disk = read_config()
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
            config = defaultdict(dict)   # items to save to global config
            parse_config_line(parts, CONFIG, config)
            if _global:
                on_disk = read_config()
                for k, v in config.items():
                    if k in on_disk and isinstance(v, dict):
                        for kk, vv in v.items():
                            on_disk[k][kk] = vv
                    else:
                        on_disk[k] = v 
                CONFIG_FILE.open('w').write(yaml.dump(on_disk))
                print('Settings saved')

