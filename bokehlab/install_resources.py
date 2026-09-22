"""Install BokehJS for classic Notebook's local/local-dev resource modes.

Copy assets instead of symlinking outside nbextensions: current Tornado rejects
such symlinks. Re-run this command after upgrading Bokeh.
"""
import argparse
from pathlib import Path
import shutil
import sys

import bokeh
from jupyter_core.paths import jupyter_data_dir


def install_resources(nbextensions_dir=None):
    root = Path(nbextensions_dir or Path(jupyter_data_dir()) / "nbextensions") / "bokeh_resources"
    if root.is_symlink():
        raise ValueError(f"Expected a directory, not a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "static"
    if destination.is_symlink():
        destination.unlink()
    source = Path(bokeh.__file__).parent / "server" / "static"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    (root / "bokeh-version.txt").write_text(bokeh.__version__ + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sys-prefix", action="store_true", help="Install into this Python environment")
    args = parser.parse_args()
    path = Path(sys.prefix) / "share/jupyter/nbextensions" if args.sys_prefix else None
    print(f"Installed Bokeh {bokeh.__version__} resources to {install_resources(path)}")
