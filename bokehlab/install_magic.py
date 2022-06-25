import os, shutil
shutil.copy('bokehlab_magic.py', os.path.expanduser('~/.ipython/profile_default/startup/'))
print('file copied ok')
