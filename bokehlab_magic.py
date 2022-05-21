from IPython.core.magic import register_line_magic

@register_line_magic
def bokehlab(line):
    "Magic equivalent to %load_ext bokehlab"
    get_ipython().run_line_magic('load_ext', 'bokehlab')
