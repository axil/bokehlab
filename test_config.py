import os
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
import bokehlab

def test1():
    # test writing config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab_config("-g resources='inline'")

        assert bokehlab.CONFIG_FILE.open().read() == yaml.dump({'resources': 'inline'})
        print('ok')

def test1a():
    # test writing config - complex key
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab_config('-g figure.width=200')

        assert bokehlab.CONFIG_FILE.open().read() == yaml.dump({'figure': {'width': 200}})
        print('ok')

def test2():
    # test reading config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab import load_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'resources': 'inline'}))
        
        load_config()

        assert bokehlab.CONFIG['resources'] == 'inline'
        print('ok')

def test2a():
    # test reading config - complex key
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'figure': {'width': 200}}))
        
        load_config()

        assert bokehlab.CONFIG.get('figure', {}).get('width') == 200
        print('ok')

def test3():
    # test setting config
    from bokehlab.bokehlab_magic import bokehlab_config

    bokehlab_config("resources='inline'")
    assert bokehlab.CONFIG['resources'] == 'inline'

    print('ok')

def test3a():
    # test setting config - complex key
    from bokehlab.bokehlab_magic import bokehlab_config

    bokehlab_config('figure.width=200')
    assert bokehlab.CONFIG['figure']['width'] == 200

    print('ok')

def test4():
    # test overriding saved config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'resources': 'inline'}))
        
        bokehlab_config("resources='local'")

        assert bokehlab.CONFIG['resources'] == 'local'
        print('ok')

def test4a():
    # test overriding saved config - complex key
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'figure': {'width': 200}}))
        
        bokehlab_config('figure.width=300')

        assert bokehlab.CONFIG.get('figure', {}).get('width') == 300
        print('ok')

def test5():
    # test deleting a key
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'resources': 'inline'}))
        
        bokehlab_config('-d resources')

        assert 'resources' not in bokehlab.CONFIG

        print('ok')

def test5a():
    # test deleting a complex key
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({
            'figure': {'width': 200, 'height': 100}}))
        
        bokehlab_config('-d figure.width')

        assert 'width' not in bokehlab.CONFIG.get('figure', {})

        print('ok')

def test6():
    # test deleting a key in config file
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'resources': 'inline'}))
        
        bokehlab_config('-g -d resources')

        on_disk = yaml.load(bokehlab.CONFIG_FILE.open().read())
        assert 'resources' not in on_disk.get('figure', {})

        print('ok')

def test6a():
    # test deleting a complex key in config file
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab.bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({
            'figure': {'width': 200, 'height': 100}}))
        
        bokehlab_config('-g -d figure.width')

        on_disk = yaml.load(bokehlab.CONFIG_FILE.open().read())
        assert 'width' not in on_disk.get('figure', {})

        print('ok')


def test7():
    # test overriding saved config
    with TemporaryDirectory() as td:
        bokehlab.CONFIG_DIR = Path(td)
        bokehlab.CONFIG_FILE = Path(td)/'bokehlab.yaml'

        from bokehlab_magic import bokehlab_config

        bokehlab.CONFIG_FILE.open('w').write(yaml.dump({'figure': {'width': 200}}))
        
        bokehlab_config('--clear --force')

        assert not os.path.exists(bokehlab.CONFIG_FILE)

        print('ok')


test1()
test2()
test3()
test4()
