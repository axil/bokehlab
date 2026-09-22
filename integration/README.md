# Notebook browser checks

The smoke test starts a temporary Jupyter server, opens a notebook in Chromium,
and checks plotting, images, tables, extension reload, linked ranges, hover,
pan/zoom, tap callbacks, and widget source updates. It uses isolated notebook
configuration and does not install resources into your normal Jupyter setup.

Modern frontends:

```sh
uv run --with playwright python -m playwright install chromium
uv run --with playwright --with 'jupyterlab>=4,<5' --with 'notebook>=7,<8' \
  python integration/notebook_smoke.py --frontend lab --resources inline
uv run --with playwright --with 'jupyterlab>=4,<5' --with 'notebook>=7,<8' \
  python integration/notebook_smoke.py --frontend notebook --resources cdn
```

Classic Notebook runs separately to avoid conflicting with Notebook 7:

```sh
uv run --isolated --no-project --python 3.12 --with-editable . \
  --with playwright --with 'notebook==6.5.7' --with 'ipykernel==6.29.5' \
  --with 'setuptools<81' --with nbformat \
  python integration/notebook_smoke.py --frontend classic --resources local
```

Repeat with `--resources inline` or `--resources local-dev`. Local modes block
external browser requests and use the real BokehLab resource installer. CDN
checks require browser access to `https://cdn.bokeh.org`.

Notebook 7 may emit an unrelated settings-schema error during frontend startup;
the harness reports it, then checks for JavaScript errors from the point where
notebook execution begins. Smoke cells run in order; the `show_df` cell exercises
Bokeh's optional table-model bundle when that feature is used.
