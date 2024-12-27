# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import subprocess
import sys

from sphinx.ext import autodoc
from docutils import nodes
from docutils.parsers.rst import Directive

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


requirements_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'requirements.txt'))
if os.path.exists(requirements_path):
    with open(requirements_path) as f:
        packages = f.readlines()
    for package in packages:
        install(package.strip())

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MinerU'
copyright = '2024, MinerU Contributors'
author = 'OpenDataLab'

# The full version, including alpha/beta/rc tags
version_file = '../../magic_pdf/libs/version.py'
with open(version_file) as f:
    exec(compile(f.read(), version_file, 'exec'))
__version__ = locals()['__version__']
# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'myst_parser',
    'sphinxarg.ext',
    'sphinxcontrib.autodoc_pydantic',
]

# class hierarchy diagram
inheritance_graph_attrs = dict(rankdir="LR", size='"8.0, 12.0"', fontsize=14, ratio='compress')
inheritance_node_attrs = dict(shape='ellipse', fontsize=14, height=0.75)
inheritance_edge_attrs = dict(arrow='vee')

autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Exclude the prompt "$" when copying code
copybutton_prompt_text = r'\$ '
copybutton_prompt_is_regexp = True

language = 'en'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
html_logo = '_static/image/logo.png'
html_theme_options = {
    'path_to_docs': 'next_docs/en',
    'repository_url': 'https://github.com/opendatalab/MinerU',
    'use_repository_button': True,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Mock out external dependencies here.
autodoc_mock_imports = [
    'cpuinfo',
    'torch',
    'transformers',
    'psutil',
    'prometheus_client',
    'sentencepiece',
    'vllm.cuda_utils',
    'vllm._C',
    # 'numpy',
    'tqdm',
]


class MockedClassDocumenter(autodoc.ClassDocumenter):
    """Remove note about base class when a class is derived from object."""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == '   Bases: :py:class:`object`':
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter

navigation_with_keys = False


# add custom directive 


class VideoDirective(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}

    def run(self):
        url = self.arguments[0]
        video_node = nodes.raw('', f'<iframe width="560" height="315" src="{url}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', format='html')
        return [video_node]

def setup(app):
    app.add_directive('video', VideoDirective)