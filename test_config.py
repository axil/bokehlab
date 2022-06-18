import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
import bokehlab

def test1():
    # test writing config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab_magic import bokehlab_config

        bokehlab_config('-g width=200')

        assert bokehlab.CONFIG_FILE.open().read() == yaml.dump({'width': 200})
        print('ok')

def test2():
    # test reading config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab import load_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'width': 200}))
        
        load_config()

        assert bokehlab.CONFIG['width'] == 200
        print('ok')

def test3():
    # test setting config
    from bokehlab_magic import bokehlab_config

    bokehlab_config('width=200')
    assert bokehlab.CONFIG['width'] == 200

    print('ok')

def test4():
    # test overriding saved config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'width': 200}))
        
        bokehlab_config('width=300')

        assert bokehlab.CONFIG['width'] == 300
        print('ok')

test1()
test2()
test3()
test4()
