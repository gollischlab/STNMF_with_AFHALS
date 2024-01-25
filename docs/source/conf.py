# Configuration file for the Sphinx documentation builder.

# -- Path setup

import os
import re
import requests
import sys
import yaml
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'stnmf'

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
    'sphinx_design',
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
    'shapely',
    'skimage',
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

stat_gen = os.path.join('_static', 'generated')
if not os.path.isdir(stat_gen):
    os.makedirs(stat_gen)


# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False
html_static_path = ['_static']
html_css_files = [
    'css/remove_links.css',
    'css/carousel_width.css',
    'css/design_wordwrap.css',
    'css/bracket_notation.css',
]

# -- Options for EPUB output

epub_show_urls = 'footnote'


# -- Citation

def get_doi(doi, mime_type):
    url = 'http://dx.doi.org/' + str(doi)
    resp = requests.get(url, headers={'accept': mime_type}, timeout=60)
    resp.encoding = 'utf-8'
    return resp.text.strip() if resp.ok else 'Reference not available'


citation = yaml.safe_load(open('../../CITATION.cff', encoding='utf-8').read())
citation_message = citation.get('message') or ''
if 'preferred-citation' in citation:
    citation = citation['preferred-citation']

doi = citation.get('doi')
if doi:
    apa = get_doi(doi, 'text/plain')
    ris = get_doi(doi, 'application/x-research-info-systems')
    bib = get_doi(doi, 'application/x-bibtex')

# Fallback until published (until DOI available)
if not doi or apa == 'Reference not available':
    # Author list with full names
    author_list_full = [author['family-names'] + ', ' + author['given-names']
                        for author in citation['authors']]
    # Author list with first initials
    author_list_init = [
        author['family-names'] + ', ' +
        ' '.join(map(lambda x: x[0]+'.',
                     author['given-names'].replace('-', ' ').split(' ')))
        for author in citation['authors']
    ]

    # APA
    if len(author_list_init) > 1:
        apa = ', '.join(author_list_init[:-1]) + ', & ' + author_list_init[-1]
    else:
        apa = author_list_init[0]
    apa += f' ({citation.get("year") or year}). {citation["title"]}'

    # RIS
    ris = 'TY  - JOUR' + '\n'
    ris += f'TI  - {citation["title"]}' + '\n'
    ris += 'AU  - ' + '\nAU  - '.join(author_list_full) + '\n'
    ris += f'PY  - {citation.get("year") or year}' + '\n'
    ris += 'ER  -' + '\n'

    # BIB
    bib = f'@{citation.get("type") or "article"}' + '{'
    bib += f'{citation["authors"][0]["family-names"]}'
    bib += f'{citation.get("year") or year},'
    bib += 'author = {' + ' and '.join(author_list_full) + '},'
    bib += r'title = {{' + citation['title'] + r'}},'
    bib += 'year = {' + str(citation.get('year') or year) + r'}}'

# Format bibtex
bib = bib.replace('ö', r'{\\"{o}}')
bib = bib.replace('ä', r'{\\"{a}}')
bib = bib.replace('ü', r'{\\"{u}}')
pos = bib.rfind('}')
if pos != -1:
    bib = bib[:pos] + '\n}\n'
bib = re.sub(r',([^=,]+)\s*=', r',\n    \1=', bib)
bib = re.sub(r'(\s{4,}[\d\w]+)\s*=\s*', r'\1 = ', bib)

with open(os.path.join(stat_gen, 'citation.ris'), 'w', encoding='utf-8') as f:
    f.write(ris)

with open(os.path.join(stat_gen, 'citation.bib'), 'w', encoding='utf-8') as f:
    f.write(bib)


# -- Substitution

rst_prolog = (
    '.. |citation-apa| replace:: ' + apa + '\n'
    '.. |citation-msg| replace:: ' + citation_message + '\n'
)
