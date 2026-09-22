import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from bokeh import events
from bokeh.core.validation import check_integrity
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.models import DatetimeAxis, GridPlot, Line, LogScale, Scatter, Span
from bokeh.plotting import figure as NativeFigure
from bokeh.resources import INLINE
from IPython.core.interactiveshell import InteractiveShell
from ipywidgets import HBox, IntSlider

import bokehlab as bl


def assert_serializable(model):
    issues = check_integrity(model.references())
    assert not issues.error
    doc = Document()
    doc.add_root(model)
    serialized = doc.to_json()
    assert serialized.content["roots"]
    assert "Bokeh" in file_html(model, INLINE)
    doc.remove_root(model)


@pytest.mark.parametrize("style, marker", [
    (".", "circle"), ("o", "circle"), ("*", "asterisk"), ("x", "x"),
    ("^", "triangle"), ("v", "triangle"), ("<", "triangle"), (">", "triangle"),
])
def test_markers(style, marker):
    p = bl.plot([1, 4, 9], style + "-", label="series", marker_size=9, get_p=True)
    assert len(p.renderers) == 2
    glyph = p.renderers[1].glyph
    assert isinstance(glyph, Scatter)
    assert glyph.marker == marker
    assert glyph.size == 9
    assert glyph.angle == bl.ANGLES.get(style, 0)
    if style == "o":
        assert glyph.fill_color == "white"
    assert len(p.legend[0].items) == 1
    assert_serializable(p)


@pytest.mark.parametrize("style, expected_style, expected_color", [
    (".-g", ".-", "g"),
    (".-bg", ".-", ["b", "g"]),
    (".-O", ".-", "O"),
])
def test_embedded_style_colors(style, expected_style, expected_color):
    y = [[1, 2], [2, 3]] if isinstance(expected_color, list) else [1, 2]
    parsed = bl.parse(y, style)
    assert [row[2] for row in parsed] == [expected_style] * len(parsed)
    assert [row[3] for row in parsed] == (
        expected_color if isinstance(expected_color, list)
        else [expected_color] * len(parsed)
    )


def test_explicit_color_overrides_embedded_style_color():
    parsed = bl.parse([1, 2], ".-g", color="r")
    assert parsed[0][2:] == (".-", "r", None)


def test_keyword_style_colors_reach_plot_glyphs():
    parsed = bl.parse([[1, 2], [2, 3]], style=".-bg")
    assert [row[2:] for row in parsed] == [
        (".-", "b", None), (".-", "g", None),
    ]
    p = bl.plot([[1, 2], [2, 3]], style=".-bg", get_p=True)
    line_colors = [r.glyph.line_color for r in p.renderers if isinstance(r.glyph, Line)]
    assert line_colors == [bl.BLUE, bl.GREEN]


@pytest.mark.parametrize("marker_size, expected_size", [(None, 4), (11, 11)])
def test_dot_marker_size_default_and_override(marker_size, expected_size):
    p = bl.plot([1, 2], ".", marker_size=marker_size, get_p=True)
    assert isinstance(p.renderers[0].glyph, Scatter)
    assert p.renderers[0].glyph.size == expected_size
    assert_serializable(p)


@pytest.mark.parametrize("style, dash", [("-", []), ("--", [6]), (":", [2, 4]), ("-.", [2, 4, 6, 4])])
def test_lines(style, dash):
    p = bl.plot([1, 2], style, get_p=True)
    assert p.renderers[0].glyph.line_dash == dash
    assert_serializable(p)


def test_sizing_and_legend():
    p = bl.plot([1, 2], label="a", width=420, height=210,
                legend_loc="top_outside", get_p=True)
    assert (p.width, p.height) == (420, 210)
    assert p.legend[0] in p.right
    assert p.toolbar_location == "above"
    assert_serializable(p)
    p = bl.Figure(width="max", height="max")
    assert p.width_policy == p.height_policy == "max"


@pytest.mark.parametrize("x", [
    [datetime(2026, 1, 1), datetime(2026, 1, 2)],
    pd.date_range("2026-01-01", periods=2, tz="UTC"),
    pd.Series(pd.date_range("2026-01-01", periods=2)),
    np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
])
def test_datetime(x):
    p = bl.plot(x, [1, 2], hover=True, vline=datetime(2026, 1, 1), get_p=True)
    assert isinstance(p.xaxis[0], DatetimeAxis)
    assert any(isinstance(r, Span) for r in p.renderers)
    assert_serializable(p)


@pytest.mark.parametrize("method, logx, logy", [
    (bl.semilogx, True, False), (bl.semilogy, False, True), (bl.loglog, True, True),
])
def test_log_axes(method, logx, logy):
    p = method([1, 10], [2, 20], get_p=True)
    assert isinstance(p.x_scale, LogScale) == logx
    assert isinstance(p.y_scale, LogScale) == logy
    assert_serializable(p)


def test_histogram_and_stem():
    h = bl.Hist([1, 1, 2, 3], bins=3, hover=True)
    assert h.histogram.sum() == 4
    assert_serializable(h.figure)
    assert_serializable(bl.stem([1, 4, 9], get_p=True))


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 3), (3, 4, 4)])
@pytest.mark.parametrize("flipud", [False, True])
def test_images(shape, flipud):
    im = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    p = bl.imshow(im, flipud=flipud, show_colorbar=len(shape) == 2, get_p=True)
    assert_serializable(p)


def test_colormap_and_linked_images():
    im = np.arange(12).reshape(3, 4)
    p = bl.imshow(im, cmap="plasma", get_p=True)
    assert p.renderers[0].glyph.color_mapper.palette == bl.mpl_cmap("plasma")
    grid = bl.imshow(im, im, get_p=True)
    assert isinstance(grid, GridPlot)
    a, b = [child[0] for child in grid.children]
    assert a.x_range is b.x_range and a.y_range is b.y_range
    assert_serializable(grid)


@pytest.mark.parametrize("stack", [bl.hstack, bl.vstack])
@pytest.mark.parametrize("merge", [True, False])
def test_layouts(stack, merge):
    a, b = bl.Plot([1, 2]), bl.Plot([2, 3])
    layout = stack(a, b, link_x=True, link_y=True, merge_tools=merge, width="max")
    assert a.x_range is b.x_range and a.y_range is b.y_range
    assert layout.sizing_mode == "stretch_width"
    assert_serializable(layout)


def test_mixed_widget_layout():
    p = bl.plot([1, 2], get_p=True)
    layout = bl.hstack(p, IntSlider())
    assert isinstance(layout, HBox)
    assert isinstance(layout.children[0], bl.BokehWidget)
    for child in layout.children:
        child.close()
    layout.close()


def test_widget_callbacks_and_source_updates(monkeypatch):
    widget, source = bl.plot([1, 2], get_ws=True)
    assert not bl.FIGURES
    changes, taps, messages = [], [], []
    widget.on_change("width", lambda attr, old, new: changes.append(new))
    widget.on_event(events.Tap, lambda event: taps.append(event.x))
    monkeypatch.setattr(widget, "send", lambda content, buffers=None: messages.append(content))
    widget._model.width = 700
    assert changes == [700]
    widget._model.document.callbacks.trigger_event(events.Tap(model=widget._model, x=1, y=2, sx=0, sy=0))
    assert taps == [1]
    source.data = {"x": [0, 1], "y": [3, 4]}
    assert any("ModelChanged" in str(message) for message in messages)
    assert "tap" in str(widget.render_bundle)
    widget.close()
    widget.close()  # ipywidgets also closes during finalization.


def test_table():
    widget, source = bl.show_df(pd.DataFrame({1: [2, 3], "b": [4, 5]}), get_ws=True)
    assert [c.field for c in widget._model.columns] == ["1", "b"]
    assert list(source.data) == ["1", "b"]
    assert widget.render_bundle
    widget.close()


def test_returns_and_figure_context():
    p, source = bl.plot([1, 2], get_ps=True)
    assert source is p.renderers[0].data_source
    assert not bl.FIGURES
    with bl.Figure() as p:
        first = bl.plot([1, 2], get_source=True)
        second = bl.plot([3, 4], get_src=True)
        assert len(p.renderers) == 2
        assert first is not second
    assert not bl.FIGURES
    assert_serializable(p)


def test_extension_reload(monkeypatch):
    shell = InteractiveShell.instance()
    monkeypatch.setattr(bl, "load", lambda: None)
    bl.load_ipython_extension(shell)
    bl.load_ipython_extension(shell)
    for event in ("pre_run_cell", "post_run_cell"):
        assert sum(hasattr(cb, "bokeh_plot_method") for cb in shell.events.callbacks[event]) == 1
    assert shell.user_ns["plot"] is bl.plot
    assert_serializable(bl.plot([1, 2], get_p=True))


def test_python_reload_preserves_native_model():
    from bokeh.model import Model
    importlib.reload(bl)
    assert Model.model_class_reverse_map["Figure"] is NativeFigure
    assert_serializable(bl.plot([1, 2], get_p=True))


@pytest.mark.parametrize("mode", ["cdn", "inline", "local", "local-dev"])
def test_resources(mode, monkeypatch):
    captured = []
    monkeypatch.setattr(bl, "output_notebook", captured.append)
    bl.load(mode)
    resources = captured[0]
    if mode.startswith("local"):
        assert all(url.startswith("/nbextensions/bokeh_resources/") for url in resources.js_files)
    elif mode == "cdn":
        assert all("3.10.0" in url for url in resources.js_files)
    else:
        assert resources.js_raw
