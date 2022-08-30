import os, shutil
from pathlib import Path

def install_magic():
    dir_path = Path(__file__).parent.absolute()
    shutil.copy(dir_path / 'bokehlab_magic.py', Path.home() / '.ipython' / 'profile_default' / 'startup')
    print('file copied ok')

if __name__ == '__main__':
    install_magic()
