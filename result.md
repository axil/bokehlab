# Bokeh 3 compatibility upgrade: result

## Implemented

- Updated the plotting layer for Bokeh 3: use the public `figure` class,
  `width`/`height`, and `scatter` markers; retain plot sizing policies, styles,
  and serializable Python-only figure customization.
- Updated widget callbacks and event subscriptions for `jupyter_bokeh` 4.x,
  including a fix that prevents widget plots from being auto-displayed into a
  second Bokeh document.
- Preserved datetime/log axes, legends, histograms, images, linked layouts,
  tables, and resource modes. Updated Matplotlib colormap access.
- Added a copy-based local BokehJS installer at
  `python -m bokehlab.install_resources`. Copying avoids the external symlink
  serving restriction in current Tornado; rerun the installer after Bokeh
  upgrades. The legacy Bokeh JavaScript patch is now a no-op.
- Made `pyproject.toml` authoritative, set Python `>=3.12` and dependencies
  `bokeh>=3.10,<4`, `jupyter_bokeh>=4.1,<5`, and `ipywidgets>=8,<9`, added the
  `uv.lock` lockfile, and retained package version `0.2.10`.
- Updated the demo and README, added regression and browser smoke tests, and
  configured CI for Python 3.12/3.13 and notebook frontend checks.

## Verification

- The final Python 3.13 suite passed: **59 tests**. Python 3.12 passed **58 tests**
  earlier in verification; the later additions were a notebook-demo metadata
  check and widget lifecycle assertion.
- Source and wheel builds succeeded; `uv lock --check`, clean wheel import/HTML
  rendering, and `git diff --check` succeeded.
- Headless browser checks passed for **JupyterLab with inline resources** and
  **classic Notebook 6 with local and local-development resources**. Checks
  exercised plotting, hover, pan/zoom, linked ranges, tap callbacks, and widget
  data updates. Local modes were tested with external browser requests blocked.

## Verification limits

- Notebook 7/CDN could not be fully validated in this environment: browser
  requests to `cdn.bokeh.org` timed out. The harness and CI matrix include this
  check, but it needs a runner with CDN access.
- Notebook 7 also emitted a frontend settings-schema error at startup here; the
  smoke harness clears startup errors before checking errors during notebook
  execution. This does not establish successful Notebook 7 browser validation.
- The demo has a static compatibility check. A full browser notebook check uses
  a dedicated smoke notebook rather than executing every demo cell.
