"""Browser smoke test. Run in an environment with Jupyter and Playwright.

Examples (install Chromium with `python -m playwright install chromium` first):
    python integration/notebook_smoke.py --frontend lab --resources inline
    python integration/notebook_smoke.py --frontend notebook --resources cdn
    python integration/notebook_smoke.py --frontend classic --resources local

Classic Notebook 6 must run in a separate environment from Notebook 7.
All notebook, configuration, and server files are confined to a temporary directory.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, build_opener

import bokeh
import jupyter_bokeh
import nbformat
from playwright.sync_api import sync_playwright
from bokehlab.install_resources import install_resources


def notebook(resources):
    return nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f'''
import numpy as np
import bokehlab as bl
from bokeh.events import Tap
from ipywidgets import Button, VBox
from IPython.display import display
bl.CONFIG['resources']['mode'] = {resources!r}
%load_ext bokehlab
import bokehlab.bokehlab_magic
%bokehlab {resources}
%blc width=450 height=280
'''), nbformat.v4.new_code_cell('''
p = plot([0, 1, 2], [1, 4, 2], '.-', hover=True, label='series', get_p=True)
p.name = 'smoke_plot'
p.title.text = 'BokehLab smoke'
q = plot([0, 1, 2], [2, 3, 1], 'x-', get_p=True)
q.name = 'linked_plot'
display(hstack(p, q, link_x=True))
'''), nbformat.v4.new_code_cell('''
w, source = plot([0, 1, 2], [1, 4, 2], 'o-', get_ws=True)
w._model.name = 'widget_plot'
w._model.title.text = 'Click plot'
def tapped(event):
    w._model.title.text = 'Tap received'
w.on_event(Tap, tapped)
button = Button(description='Update source')
def update(_):
    source.data = dict(x=[0, 1, 2], y=[3, 1, 4])
    w._model.title.text = 'Source updated'
button.on_click(update)
box = VBox([button, w])
display(box)
'''), nbformat.v4.new_code_cell('''
imshow(np.arange(12).reshape(3, 4), show_colorbar=True)
'''), nbformat.v4.new_code_cell('''
show_df(__import__('pandas').DataFrame({'a': [1, 2], 'b': [3, 4]}))
print('BOKEHLAB_SMOKE_READY')
''')], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def plot_point(page, name, x, y):
    return page.evaluate("""({name, x, y}) => {
        const model = Bokeh.documents.map(d => d.get_model_by_name(name)).find(Boolean);
        const view = Bokeh.index.find_one(model);
        const el = view.canvas_view.events_el;
        el.scrollIntoView({block: 'center'});
        const rect = el.getBoundingClientRect();
        return {x: rect.x + view.frame.x_scale.compute(x),
                y: rect.y + view.frame.y_scale.compute(y)};
    }""", {"name": name, "x": x, "y": y})


def check_interactions(page):
    point = plot_point(page, "smoke_plot", 1, 4)
    page.mouse.move(point["x"], point["y"])
    page.locator(".bk-Tooltip").filter(has_text="series").first.wait_for(timeout=10000)
    initial = page.evaluate("Bokeh.documents.map(d => d.get_model_by_name('smoke_plot')).find(Boolean).x_range.start")
    point = plot_point(page, "smoke_plot", 1, 2)
    page.mouse.move(point["x"], point["y"])
    page.mouse.down()
    page.mouse.move(point["x"] + 50, point["y"], steps=10)
    page.mouse.up()
    page.wait_for_function("start => Bokeh.documents.map(d => d.get_model_by_name('smoke_plot')).find(Boolean).x_range.start !== start", arg=initial)
    width = page.evaluate("""() => {
        const p = Bokeh.documents.map(d => d.get_model_by_name('smoke_plot')).find(Boolean);
        return p.x_range.end - p.x_range.start;
    }""")
    page.mouse.wheel(0, -150)
    page.wait_for_function("""width => {
        const p = Bokeh.documents.map(d => d.get_model_by_name('smoke_plot')).find(Boolean);
        return Math.abs(p.x_range.end - p.x_range.start - width) > 0.001;
    }""", arg=width)
    point = plot_point(page, "widget_plot", 1, 2)
    page.mouse.click(point["x"], point["y"])
    page.wait_for_function("Bokeh.documents.some(d => d.get_model_by_name('widget_plot')?.title.text === 'Tap received')", timeout=15000)


def execute_lab_cell(page, index):
    """Execute one Lab cell and wait until its kernel execution completes."""
    cell = page.locator(".jp-Notebook .jp-CodeCell").nth(index)
    cell.locator(".jp-InputArea-editor").click()
    page.keyboard.press("Shift+Enter")
    page.wait_for_function("""index => {
        const cell = document.querySelectorAll('.jp-Notebook .jp-CodeCell')[index];
        if (!cell || cell.classList.contains('jp-mod-running')) return false;
        const prompt = cell.querySelector('.jp-InputPrompt');
        const text = prompt?.textContent?.trim() ?? '';
        return text !== '' && !text.startsWith('[ ]');
    }""", arg=index, timeout=60000)


def run(frontend, resources):
    with tempfile.TemporaryDirectory(prefix="bokehlab-browser-") as directory:
        root = Path(directory)
        nbformat.write(notebook(resources), root / "smoke.ipynb")
        env = os.environ.copy()
        env.update(JUPYTER_CONFIG_DIR=str(root / "config"),
                   JUPYTER_DATA_DIR=str(root / "data"),
                   JUPYTER_RUNTIME_DIR=str(root / "runtime"),
                   IPYTHONDIR=str(root / "ipython"))
        widget_prefix = Path(jupyter_bokeh.__file__).resolve().parents[4]
        env["JUPYTER_PATH"] = str(widget_prefix / "share/jupyter") + os.pathsep + env.get("JUPYTER_PATH", "")
        # Use the current test interpreter, not a system-wide kernelspec.
        kernel_dir = root / "data/kernels/python3"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3", "language": "python",
        }))
        if frontend == "classic":
            config = root / "config/nbconfig"
            config.mkdir(parents=True)
            (config / "notebook.json").write_text(json.dumps({"load_extensions": {
                "jupyter-js-widgets/extension": True, "jupyter_bokeh/extension": True,
            }}))
        if resources.startswith("local"):
            install_resources(root / "data/nbextensions")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        app = "NotebookApp" if frontend == "classic" else "ServerApp"
        module = "jupyterlab" if frontend == "lab" else "notebook"
        command = [sys.executable, "-m", module, "--no-browser", f"--port={port}",
                   "--ip=127.0.0.1", f"--{app}.token=bokehlab-smoke",
                   f"--{app}.password=", f"--{app}.allow_root=True"]
        logfile = (root / "server.log").open("w+")
        process = subprocess.Popen(command, cwd=root, env=env, stdout=logfile, stderr=logfile)
        base = f"http://127.0.0.1:{port}"
        opener = build_opener(ProxyHandler({}))
        try:
            for _ in range(120):
                try:
                    with opener.open(base + "/api?token=bokehlab-smoke", timeout=1):
                        break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError("Notebook server exited")
                    time.sleep(0.25)
            else:
                raise RuntimeError("Notebook server failed to start")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
                page = browser.new_page(viewport={"width": 1400, "height": 1100})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("pageerror", lambda error: print("Page error:", error.stack, flush=True))
                page.on("console", lambda message: print("browser:", message.text, flush=True)
                        if message.type == "error" else None)
                page.on("requestfailed", lambda request: print("Request failed:", request.url, request.failure, flush=True))
                if resources.startswith("local"):
                    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base)
                               or route.request.url.startswith("data:") else route.abort())
                path = "/lab/tree/smoke.ipynb" if frontend == "lab" else "/notebooks/smoke.ipynb"
                page.goto(base + path + "?token=bokehlab-smoke")
                if frontend == "classic":
                    page.wait_for_function("window.Jupyter && Jupyter.notebook && Jupyter.notebook.kernel && Jupyter.notebook.kernel.is_connected()")
                    page.evaluate("Jupyter.notebook.execute_cells([0])")
                else:
                    page.locator(".jp-Notebook .jp-CodeCell").first.wait_for(timeout=60000)
                    # Ignore frontend startup errors before any BokehLab code runs.
                    errors.clear()
                    execute_lab_cell(page, 0)
                # Bokeh's tables model is loaded on demand. Don't wait for its
                # optional bundle here; the show_df cell below triggers it.
                if frontend == "classic":
                    page.evaluate("Jupyter.notebook.execute_cells([1, 2, 3, 4])")
                else:
                    for index in range(1, 5):
                        execute_lab_cell(page, index)
                print(f"Executing {frontend}/{resources}", flush=True)
                try:
                    page.locator(".jp-OutputArea-output, .output_area").filter(
                        has_text="BOKEHLAB_SMOKE_READY").first.wait_for(timeout=30000)
                except Exception:
                    print(page.locator("body").inner_text()[-6000:], file=sys.stderr)
                    raise
                page.wait_for_function("window.Bokeh && Bokeh.documents.some(d => d.get_model_by_name('smoke_plot'))", timeout=30000)
                page.wait_for_function("Bokeh.documents.some(d => d.get_model_by_name('widget_plot'))", timeout=15000)
                assert not page.locator(".jp-OutputArea-error, .output_error").count()
                check_interactions(page)
                # The browser model must update after a real kernel callback.
                try:
                    page.get_by_role("button", name="Update source", exact=True).click(timeout=15000)
                except Exception:
                    print(page.locator("body").inner_text()[-7000:], file=sys.stderr)
                    print("Page errors:", errors, file=sys.stderr)
                    raise
                page.wait_for_function("Bokeh.documents.some(d => d.get_model_by_name('widget_plot')?.title.text === 'Source updated')", timeout=30000)
                result = page.evaluate("""() => {
                    const p = Bokeh.documents.map(d => d.get_model_by_name('smoke_plot')).find(Boolean);
                    const q = Bokeh.documents.map(d => d.get_model_by_name('linked_plot')).find(Boolean);
                    return {version: Bokeh.version, linked: p.x_range === q.x_range};
                }""")
                assert result["version"] == bokeh.__version__
                assert result["linked"]
                assert page.locator("canvas").count() > 0
                assert not errors, errors
                print(f"PASS {frontend} / {resources}: {result}")
                browser.close()
        except Exception:
            logfile.flush()
            print((root / "server.log").read_text()[-8000:], file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            logfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=["lab", "notebook", "classic"], required=True)
    parser.add_argument("--resources", choices=["inline", "cdn", "local", "local-dev"], default="inline")
    args = parser.parse_args()
    run(args.frontend, args.resources)
