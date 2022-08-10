import os
from collections import defaultdict
from copy import deepcopy

import yaml
from IPython.core.display import display, HTML


def parse_config_line(parts, CONFIG, config=None, verbose=True):
    from bokehlab import CONFIG, CONFIG_DIR, CONFIG_FILE, CONFIG_SECTIONS, load_config, RESOURCE_MODES    ##
    if '-v' in parts or '--verbose' in parts:
        verbose = True
    for part in parts:
        if '=' not in part:
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
    from bokehlab import CONFIG_DIR, CONFIG_FILE
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir()
    if CONFIG_FILE.exists():
        on_disk = yaml.load(CONFIG_FILE.open().read(), yaml.SafeLoader)
    else:
        on_disk = {}
    return on_disk

def config_bokehlab(line, cell=None):
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
    from bokehlab import CONFIG, CONFIG_DIR, CONFIG_FILE, CONFIG_SECTIONS, load_config, RESOURCE_MODES
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
            _global = False
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
