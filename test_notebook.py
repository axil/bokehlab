from pathlib import Path

import nbformat


def test_demo_notebook_is_current_bokeh_compatible():
    notebook = nbformat.read(Path(__file__).with_name("demo.ipynb"), as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "%load_ext bokehlab" in source
    assert "%load_ext bokeh_plot" not in source
    assert "plot_width=" not in source
    assert "plot_height=" not in source
    assert "p.circle(" not in source
