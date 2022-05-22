#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bokehlab",
    version="0.2.2",
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/bokehlab',
    description="Interactive plotting with familiar syntax in jupyter notebooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['bokeh', 'pandas', 'matplotlib'],
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
