import os
from pathlib import Path
from shutil import move

import bokeh

os.chdir(Path(bokeh.__file__).parent / 'server' / 'static' / 'js')
txt = open('bokeh.min.js', newline='').read()
context = 'n=(0,u._resolve_root_elements)(e);(0,f.add_document_standalone)(t,o,n)'
fix = ";n[0].removeAttribute('id')"
if context+fix in txt:
    print('Already patched')
elif context in txt:
    move('bokeh.min.js', 'bokeh.min.js.bak')
    txt = txt.replace(context, context + fix, 1)
    with open('bokeh.min.js', 'w', newline='') as f:
        f.write(txt)
    print('Patched ok')
else:
    print('Patch not appliable')
