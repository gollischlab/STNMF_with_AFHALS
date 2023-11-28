# Configuration file for the Sphinx documentation builder.

# -- Path setup

import os
import re
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = name = 'stnmf'

# Load version from stnmf/__init__.py
res = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                open(f'../../{project}/__init__.py', encoding='utf8').read())
version = res.group(1)
release = version

# Load copyright and authors from LICENSE
res = re.search(r'Copyright\s*\(c\)\s*(.*)\n',
                open('../../LICENSE', encoding='utf8').read())
year, author = res.group(1).split('  ')  # Assume two spaces after years
copyright = year + ', ' + author

# -- General configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_carousel.carousel',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'inherited-members': True,
    'show-inheritance': True,
    'ignore-module-all': True,
    'class-doc-from': 'class',
}
autodoc_mock_imports = [
    'cycler',
    'matplotlib',
    'mpl_toolkits',
    'numpy',
    'scipy',
    'skimage',
    'shapely',
    'tqdm',
]

copybutton_only_copy_prompt_lines = True
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "  # noqa: E501
copybutton_prompt_is_regexp = True

napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "color": ":ref:`color <colors_def>`",
    "class": ":class:`class <type>`",
    "array_like": ":term:`array_like`",
}
napoleon_custom_sections = [
    ('Factorization Args', 'params_style'),
    ('Callback Args', 'params_style'),
]

mathjax_path = ('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/'
                'MathJax.js?config=TeX-MML-AM_CHTML')

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'attrs': ('https://attrs.org/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = [
    'css/remove_links.css',
    'css/carousel_width.css',
]

# -- Options for EPUB output

epub_show_urls = 'footnote'
