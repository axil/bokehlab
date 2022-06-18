from IPython.core.magic import register_line_magic

@register_line_magic
def bokehlab(line):
    """
    Magic equivalent to %load_ext bokehlab. Injects keywords like 'plot'
    into global namespace.
    """
    get_ipython().run_line_magic('load_ext', 'bokehlab')

@register_line_magic
def bokehlab_config(line):
    "Magic equivalent to %load_ext bokehlab"
    parts = line.split()
    for part in parts:
        if part in ('-g', '--global'):
            print('global')
        elif '=' in part:
            k, v = part.split('=', 1)
            print(k, '=', v)

