# Debugging Bokeh rendering with a headless browser

Playbook for investigating "the plot looks wrong in the browser" problems without a
notebook server. It was developed while fixing the `imshow` view-range bug (Bokeh 3.10,
see `journal/2026-11-19.md`) and is generally useful for any Bokeh/BokehLab rendering
issue. Concrete scripts from that investigation are not committed, but every step below
is copy-pasteable.

## 1. Build a standalone reproduction

No Jupyter needed — serialize the exact figure to a static page:

```python
import numpy as np
from bokeh import embed, resources
from bokeh.plotting import figure

p = figure(width=330, height=300, y_range=[220, -20])
# ... reproduce the exact glyph setup ...
html = embed.file_html(p, resources.INLINE, "repro")
open("repro.html", "w", encoding="utf-8").write(html)   # encoding matters on Windows
```

Tips:

- Use `resources.INLINE` so the page works from `file://` with no server and no CDN.
- Encode coordinates in the test image so orientation/clipping is visible at a glance:
  a gradient like `value = (row/H + col/W) / 2` makes "which corner is dark" reveal both
  clipping (image cut off) and flips (gradient inverted).
- To tell a wrapper bug from a library bug, **bisect**: build the same figure with raw
  Bokeh only (no BokehLab classes). If raw Bokeh misbehaves too, it's a Bokeh issue —
  work around it in the wrapper rather than "fixing" the wrapper.

## 2. Render it headlessly with Playwright (no browser download)

Playwright can drive an already-installed Chrome/Edge, so `playwright install` is not
required. `uv run --with` keeps the machine's Python clean:

```python
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome")   # or "msedge"
        page = await browser.new_page(viewport={"width": 1000, "height": 1600})
        await page.goto(Path("repro.html").resolve().as_uri())
        await page.wait_for_selector("div.bk-Figure", timeout=15000)
        for el in await page.query_selector_all("div.bk-Figure"):
            await el.screenshot(path="shot.png")               # one PNG per plot
        await browser.close()

asyncio.run(main())
```

Run with: `uv run --with playwright --with pillow python script.py`

Quick pass/fail without eyeballing: open the PNG with Pillow and compute the bounding
box of non-background pixels (dark/saturated mask) — a clipped image shows a tiny bbox
flush against the frame; a correct render covers ~90% of it (with padding).

## 3. Bokeh 3 gotcha: the plot DOM lives in shadow DOM

Bokeh 3 isolates each root in a shadow root. Playwright selectors pierce shadow DOM
automatically (`page.query_selector_all("div.bk-Figure")` works), but **in-page
JavaScript does not** — `document.querySelectorAll` returns ~15 light-DOM nodes and no
figures. Walk shadow roots manually inside `page.evaluate`:

```js
function* walk(root) {
    for (const el of root.querySelectorAll("*")) {
        yield el;
        if (el.shadowRoot) yield* walk(el.shadowRoot);
    }
}
const figs = [...walk(document)].filter(el =>
    typeof el.className === "string" && el.className.includes("bk-Figure"));
```

## 4. Read BokehJS internals at runtime

`window.Bokeh.index` is a `ViewManager` (proxy). Use `all_views()` / `get_by_id(id)`
to reach live views, then walk to the glyph view and dump what the renderer actually
computed — this is how the off-screen paint coordinates were pinned down:

```js
const views = [...window.Bokeh.index.all_views()];
const plotView = views.find(v => v.shadow_el === figEl);
const rendererView = [...plotView.renderer_views.values()]
    .find(v => v.model.type === "GlyphRenderer");
const gv = rendererView.glyph_view;
return {
    origin: gv.model.origin, anchor: gv.model.anchor,
    sy: Array.from(gv.sy), sdh: Array.from(gv.sdh),          // mapped screen coords
    data_y: rendererView.model.data_source.data.y,           // what was actually sent
    y_range: [plotView.model.y_range.start, plotView.model.y_range.end],
    frame: [plotView.frame.bbox.x0, plotView.frame.bbox.y0,
            plotView.frame.bbox.x1, plotView.frame.bbox.y1],
};
```

`sy`/`sdh` are the glyph's screen position/size — comparing them against `frame.bbox`
immediately shows whether the glyph lands inside the visible area. Also capture
console/page errors (`page.on("console")`, `page.on("pageerror")`); a failed embed can
leave an apparently blank page.

## 5. Run a small matrix to find the correct recipe

When several interacting flags could be responsible (here: range direction, data
flip, glyph `origin`), render all combinations in one page, screenshot each, and pick
the one that is correct. Empirical beats theoretical: hand-simulating BokehJS's paint
transforms predicted the wrong outcome twice, the matrix settled it in one run.

## Pitfalls hit during the investigation

- **Stale installed package**: `python script.py` run from a scratch directory
  imported an old `bokehlab` from site-packages, not the repo checkout — the fix
  "didn't work" until `PYTHONPATH=<repo>` was set. Always print
  `bokehlab.__file__` (or the module under test) before trusting a render.
- **Windows encoding**: write HTML with `encoding="utf-8"` (inline Bokeh resources
  contain non-CP1251 characters).
- **`file://` URLs**: pass `Path("repro.html").resolve().as_uri()` to `page.goto`,
  not a bare path.
