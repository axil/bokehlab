#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bokeh_plot",
    version="0.1.9",
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/bokeh-plot',
    description="Matlab-inspired call syntax for bokeh plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['bokeh'],
    packages=['bokeh_plot'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT License",
    zip_safe=False,
    keywords=['bokeh', 'jupyter', 'notebook', 'plot'],
)
