# Bokeh 3 compatibility upgrade: specification

## Goal

Upgrade BokehLab to work with current Bokeh while preserving its familiar plotting
syntax, notebook behavior, and existing plotting features. The target is Bokeh
3.10, with no Bokeh 2 compatibility layer.

## Compatibility and packaging

- Require Python 3.12 or newer, Bokeh 3.10 or newer within the 3.x series, and
  `jupyter_bokeh` 4.x.
- Make `pyproject.toml` the authoritative package metadata and dependency
  declaration. Preserve the existing package version until a release is requested.
- Validate Python 3.12 and 3.13, and provide reproducible development dependencies
  and a lockfile.

## Bokeh API and notebook behavior

- Migrate removed or changed Bokeh APIs while preserving plot, marker, axis,
  sizing, legend, image, histogram, layout, table, and data-return behavior.
- Keep CDN, inline, local, and local-development resource modes. Support
  JupyterLab 4, Notebook 7, and classic Notebook 6; document frontend-specific
  setup and resource limitations.
- Preserve widget rendering, Python callbacks, notebook extension reloads, and
  local offline resources. Do not patch files inside the installed Bokeh package.

## Acceptance checks

- Test plotting behavior and Bokeh model validation/serialization, plus notebook
  rendering and interactions: hover, pan, zoom, linked ranges, tap callbacks,
  and source updates.
- Build source and wheel distributions and verify a clean installation.
- Update the demo and user documentation for the supported Python, Bokeh, and
  Jupyter versions.
