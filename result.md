# Result: bokehlab upgraded to bokeh 3.x

bokehlab now runs on **bokeh 3.10.0 + jupyter-bokeh 4.1.0** (previously 2.4.3 / 3.0.4,
which could not even import under numpy 2.5 due to the `np.bool8` removal).
Released as **version 0.3.0**; the package was reinstalled from source so the installed
metadata matches the new pins.

## Changes

### `bokehlab/__init__.py`

- `Figure` now subclasses `bokeh.plotting.figure` (the class is lowercase in bokeh 3; the
  old `bokeh.plotting.figure.Figure` import path is gone).
- Replaced the removed `__subtype__` mechanism with
  `__view_model__ = "Figure"` / `__view_module__ = "bokeh.plotting._figure"`, so the
  subclass serializes as the BokehJS Figure model. Added a targeted
  `warnings.filterwarnings` for bokeh 3's "Duplicate qualified model definition"
  `BokehUserWarning` (imported from `bokeh.util.warnings`) that fires on import and on
  `reload_ext`.
- Deleted `bm.Model.model_class_reverse_map.pop('BokehlabFigure', None)` — a no-op in
  bokeh 3 (the property returns a copy).
- Deleted the `width`→`plot_width` / `height`→`plot_height` conversion; bokeh 3 accepts
  `width`/`height` directly and rejects `plot_width`/`plot_height`.
- Markers `.`, `o`, `*`, `^v<>` now render via `p.scatter(marker=...)` — `circle(size=...)`,
  `asterisk()` and `triangle()` are deprecated since bokeh 3.4, and `circle()` without an
  explicit size failed serialization entirely (unset `radius`). `.` markers get an explicit
  default `size=4` (matching the bokeh 2 default).
- `mpl_cmap`: `cm.get_cmap(name)` → `matplotlib.colormaps[name]` (removed in matplotlib 3.9;
  installed matplotlib is 3.11).
- `__version__ = '0.3.0'`.

### `bokehlab/palettes.py`

- Same colormap fix in `get_mpl`; dropped the unused `matplotlib.cm` import.

### Packaging

- `setup.py`: version 0.3.0; `install_requires` = `['bokeh>=3', 'jupyter-bokeh>=4',
  'matplotlib', 'pyyaml', 'pandas']` (dropped `bokeh<3`, `jupyter-bokeh<=3.0.4` and the
  duplicate `jupyter_bokeh` entry).
- `requirements.txt`: `bokeh<3` → `bokeh>=3`.
- `whatsnew.txt`: added the 0.3.0 entry.
- `.gitignore`: added `bokehlab.egg-info/`.

## Pre-existing bugs fixed along the way

These were broken at HEAD independent of bokeh 3, but surfaced during verification:

- **Embedded color letters in style strings** (`bokehlab/__init__.py`, `parse()`):
  documented syntax like `plot(x, y, '.-g')` and `plot([y1, y2], '.-bg')` raised
  `ValueError: Unsupported plot style` — the 0.2.3 parser rewrite never split color letters
  out of the style spec. `parse()` now extracts them (a leading marker character such as
  `o` is preserved), with regression tests added to `tests.py`.
- **`config.py` typo:** `k = 'resources.mode' + k` turned `%blc resources=inline` into a
  bogus `resources.moderesources` key. Fixed to `k = 'resources.mode'`.
- **`config.py` `-g -d` no-op:** the delete branch force-reset `_global = False`, so
  `%blc -g -d key` never wrote the deletion to disk. Fixed.
- **`test_config.py` was silently writing to the real `~/.bokeh/bokehlab.yaml`:** it
  patched `bokehlab.CONFIG_FILE`, which ceased to exist when config moved to `config.py`.
  Tests now patch `bokehlab.config.CONFIG_DIR/CONFIG_FILE`, and stale `resources`
  expectations were updated to the dict form (`{'resources': {'mode': ...}}`). The garbage
  written to the real user config during a failing run was removed (the file did not exist
  beforehand).
- **`tests.py`:** removed a stale `AUTOCOLOR` reference (commented out of `__init__.py`
  long ago) and updated one assertion that predated intentional string-label broadcasting
  (`label=['y']` instead of `label='y'` for the mismatch error case).

## Verification results

- `python tests.py` — all parser tests + exception tests pass.
- `ipython test_config.py` — all 4 config tests pass; no user config file touched.
- Serialization smoke test (temporary script, removed afterwards): 39 HTML files saved via
  `bokeh.io.save` covering plot (markers, legend incl. `outside`, hover with `idx`,
  vline/hline, labels, dashed/dotted/dotdash styles), pandas/dict inputs, stem,
  semilogx/semilogy/loglog, hist with hover, datetime x-axes (naive/tz-aware lists,
  DatetimeIndex, Series, Timestamps, df index, datetime vline), imshow (float, uint8,
  RGB, RGBA, colorbar, matplotlib palette, flipud, linked pair via gridplot, hover),
  `figure()` + FIGURES autoshow path, hstack/vstack with linked axes and merged tools,
  show_df, and BokehWidget (`render_bundle` non-empty). Saved JSON confirmed to contain
  `"name":"Figure"` model references.
- End-to-end notebook: a fresh notebook (7 cells: `%load_ext bokehlab`, plot, multi-plot
  with hover, imshow, hist, show_df, stem, loglog) executed via
  `jupyter nbconvert --execute` — zero errors, every cell produced bokeh HTML output.

## Left for the user

- **Visual check in a real browser/JupyterLab session** — everything verifiable headless
  passes, but rendering was not eyeballed.
- `demo.ipynb` still starts with `%load_ext bokeh_plot` (pre-rename extension name) and
  needs a refresh before it can run.
- README documents `plot(x, y1, '.-', x, y2, '.-g')` (multiple x/y/style triples); the
  current parser rejects more than 5 positional arguments. Not restored — pending decision.
- `fix_copy_paste.py` targets bokeh 2.x `bokeh.min.js` internals; under bokeh 3.10 it
  reports "Patch not appliable". Whether the underlying copy/paste issue still exists in
  bokeh 3 was not investigated.
- pytest was installed into the environment (needed by `tests.py`'s exception tests).
