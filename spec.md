# Spec: Upgrade bokehlab to bokeh 3.x

## Goal

Make bokehlab compatible with the latest bokeh release (3.10.0) and the matching
jupyter-bokeh (4.1.0). The previous pins (`bokeh<3`, `jupyter-bokeh<=3.0.4`) target bokeh 2.x,
which no longer imports under numpy 2.x (`np.bool8` removal). jupyter-bokeh 4.x requires
`bokeh ==3.*` and `ipywidgets ==8.*`, so the two dependencies are upgraded together.

Scope: bokeh 3.x only — no dual bokeh 2/3 compatibility (import paths and the jupyter-bokeh
pairing make dual support not worthwhile).

## 1. Environment

```
pip install -U "bokeh>=3.10" "jupyter-bokeh>=4.1"
```

## 2. `bokehlab/__init__.py`

1. **Figure class import.** `from bokeh.plotting.figure import Figure` no longer exists; in
   bokeh 3 the class is lowercase: `from bokeh.plotting import figure as BokehFigure`.

2. **Figure subclass declaration.** `__subtype__` was removed in bokeh 3. A pure-Python
   subclass with no new properties declares the built-in view model instead:
   ```python
   class Figure(BokehFigure):
       __view_model__ = "Figure"
       __view_module__ = "bokeh.plotting._figure"
   ```
   This re-registers the qualified name "Figure", which bokeh 3 flags with a
   `BokehUserWarning` ("Duplicate qualified model definition") at class-definition time and
   on every `reload_ext`. Suppress exactly that warning module-wide:
   ```python
   import warnings
   from bokeh.util.warnings import BokehUserWarning
   warnings.filterwarnings('ignore', category=BokehUserWarning,
                           message='Duplicate qualified model definition')
   ```

3. **Delete the reload hack** `bm.Model.model_class_reverse_map.pop('BokehlabFigure', None)`.
   In bokeh 3 `model_class_reverse_map` returns a copy, so popping is a no-op; with
   `__subtype__` gone nothing registers under 'BokehlabFigure' anyway. The warning filter
   in step 2 covers reloads.

4. **`plot_width`/`plot_height` removed in bokeh 3.** Delete the block renaming
   `width`→`plot_width` / `height`→`plot_height`; bokeh 3's `figure()` takes `width`/`height`
   directly and raises `AttributeError` on unknown kwargs.

5. **Markers via `scatter`.** `circle(size=...)`, `asterisk()`, `triangle()` are deprecated
   in bokeh 3.4+ (`circle()` without an explicit size breaks outright: unset `radius`).
   Route all marker styles through `p.scatter(marker='circle'|'asterisk'|'triangle', ...)`
   and give `.` markers an explicit default `size=4` (the bokeh 2 default).

6. **matplotlib colormaps.** `matplotlib.cm.get_cmap` was removed in matplotlib 3.9; use
   `matplotlib.colormaps[name]` in `mpl_cmap`.

Verified unchanged against bokeh 3.10 / jupyter-bokeh 4.1 sources (no action needed):
glyph methods (`line/segment/quad/image/image_rgba`), `legend_label`, legend splat access
(`p.legend.location`, `p.legend[0]`, `add_layout(..., 'right')`), HoverTool `$name`,
`@x{%F}` and `formatters={'@x': 'datetime'}`, `Span`, `DataTable`, `gridplot/row/column`
(incl. `merge_tools`, `sizing_mode`), `bokeh.layouts.LayoutDOM`, `output_notebook` /
`push_notebook` / `show(notebook_handle=True)`, `Resources('server'|'server-dev',
root_url=...)`, `DataRange1d.flipped`, `LinearColorMapper`/`ColorBar`, all `bokeh.models`
imports. jupyter_bokeh 4.1 keeps `BokehModel(model)`, `render_bundle`, `_model_to_traits`;
bokeh 3.10 keeps `Model._update_event_callbacks`, so `BokehWidget` is unchanged. Bokeh 3's
`HasProps.__setattr__` requires underscore-prefixed Python-side attributes — all of
bokehlab's (`_hover`, `_legend_location`, ...) already comply.

## 3. `bokehlab/palettes.py`

`get_mpl`: `cm.get_cmap(name)` → `matplotlib.colormaps[name]`.

## 4. Packaging / metadata

- `setup.py`: version `0.2.10` → `0.3.0`; `install_requires` →
  `['bokeh>=3', 'jupyter-bokeh>=4', 'matplotlib', 'pyyaml', 'pandas']`
  (drops `bokeh<3`, `jupyter-bokeh<=3.0.4`, duplicate `jupyter_bokeh`).
- `bokehlab/__init__.py`: `__version__ = '0.3.0'`.
- `requirements.txt`: `bokeh<3` → `bokeh>=3`.
- `whatsnew.txt`: add the 0.3.0 entry.
- Reinstall from source (`pip install .`) so installed metadata matches.

## 5. Verification

1. `python tests.py` — parser + exception tests.
2. `ipython test_config.py` — config magic tests (needs an IPython context).
3. Serialization smoke test (main risk: the Figure subclass serializing as BokehJS
   "Figure"): temp script building every feature with `get_p=True` and saving via
   `bokeh.io.save` — plot (markers/legend/hover/idx), stem, hist, semilogx/semilogy/loglog,
   imshow (float/uint8/RGB/RGBA/colorbar/mpl palette/linked pair/flipud/hover), show_df,
   hstack/vstack, datetime axes with datetime vline/hline, `figure()` + FIGURES autoshow,
   BokehWidget.
4. End-to-end notebook run: a temp notebook executed via `jupyter nbconvert --execute`
   covering `%load_ext bokehlab`, plot, imshow, hist, show_df, stem, loglog.

## Out of scope / known gaps

- `bokehlab/fix_copy_paste.py` patches a bokeh 2.x `bokeh.min.js` string; under bokeh 3 it
  prints "Patch not appliable". Left as-is.
- `demo.ipynb` still references the pre-rename `%load_ext bokeh_plot`.
- README's multi-triple syntax `plot(x, y1, '.-', x, y2, '.-g')` is not supported by the
  current parser (>5 positional args).
