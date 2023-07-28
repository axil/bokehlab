#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bokehlab",
    version="0.2.10",
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/bokehlab',
    description="Interactive plotting with familiar syntax in Jupyter notebooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['bokeh<3', 'jupyter-bokeh<=3.0.4', 'matplotlib', 'pyyaml', 'jupyter_bokeh', 'pandas'],
    packages=['bokehlab'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT License",
    zip_safe=False,
    keywords=['bokeh', 'jupyter', 'notebook', 'plot', 'interactive'],
)
