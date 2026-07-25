"""Configuration file for the Sphinx documentation builder."""

import re
import tomllib
from pathlib import Path

import requests
import yaml

# Project information
project = "stnmf"

# File paths
path_root = Path(__file__).parents[2]
path_project = path_root / "pyproject.toml"
path_license = path_root / "LICENSE"
path_citation = path_root / "CITATION.cff"

# Read files
pyproject = tomllib.loads(path_project.read_text("utf-8"))
license = path_license.read_text("utf-8")
citation = path_citation.read_text("utf-8")

# Load project version
version = release = pyproject["project"]["version"]

# Load copyright and authors from LICENSE
res = re.search(r"Copyright\s*\(c\)\s*((?:\d{4}[\s,-]+)+)(\w.*)\n", license)
assert res is not None, "Copyright notice in LICENSE is mal-formatted"
years = res.group(1).strip()
year = years[:4]
author = res.group(2)
copyright = years + ", " + author


# -- General configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx_carousel.carousel",
    "sphinx_design",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
    "class-doc-from": "class",
}

autodoc_mock_imports = [
    "cycler",
    "matplotlib",
    "mpl_toolkits",
    "numpy",
    "scipy",
    "shapely",
    "skimage",
    "tqdm",
]

copybutton_only_copy_prompt_lines = True
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
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
    ("Factorization Args", "params_style"),
    ("Callback Args", "params_style"),
]

mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/"
    "MathJax.js?config=TeX-MML-AM_CHTML"
)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "attrs": ("https://attrs.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

stat_gen = Path("_static", "generated")
stat_gen.mkdir(parents=True, exist_ok=True)


# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = [
    "css/remove_links.css",
    "css/carousel_width.css",
    "css/design_wordwrap.css",
    "css/bracket_notation.css",
]

# -- Options for EPUB output

epub_show_urls = "footnote"


# -- Citation


def get_doi(doi, mime_type):
    url = "http://dx.doi.org/" + str(doi)
    resp = requests.get(url, headers={"accept": mime_type}, timeout=60)
    resp.encoding = "utf-8"
    return resp.text.strip() if resp.ok else "Reference not available"


citation = yaml.safe_load(citation)
citation_message = citation.get("message") or ""
if "preferred-citation" in citation:
    citation = citation["preferred-citation"]

doi = citation.get("doi")
if doi:
    apa = get_doi(doi, "text/plain")
    ris = get_doi(doi, "application/x-research-info-systems")
    bib = get_doi(doi, "application/x-bibtex")

# Fallback until published (until DOI available)
if not doi or apa == "Reference not available":
    # Author list with full names
    author_list_full = [
        author["family-names"] + ", " + author["given-names"]
        for author in citation["authors"]
    ]
    # Author list with first initials
    author_list_init = [
        author["family-names"]
        + ", "
        + " ".join(
            x[0] + "." for x in author["given-names"].replace("-", " ").split(" ")
        )
        for author in citation["authors"]
    ]

    # APA
    if len(author_list_init) > 1:
        apa = ", ".join(author_list_init[:-1]) + ", & " + author_list_init[-1]
    else:
        apa = author_list_init[0]
    apa += f" ({citation.get('year') or year}). {citation['title']}"

    # RIS
    ris = "TY  - JOUR" + "\n"
    ris += f"TI  - {citation['title']}" + "\n"
    ris += "AU  - " + "\nAU  - ".join(author_list_full) + "\n"
    ris += f"PY  - {citation.get('year') or year}" + "\n"
    ris += "ER  -" + "\n"

    # BIB
    bib = f"@{citation.get('type') or 'article'}" + "{"
    bib += f"{citation['authors'][0]['family-names'].replace(' ', '')}"
    bib += f"{citation.get('year') or year},"
    bib += "author = {" + " and ".join(author_list_full) + "},"
    bib += r"title = {{" + citation["title"] + r"}},"
    bib += "year = {" + str(citation.get("year") or year) + r"}}"
else:
    # Edge case: bioRxiv is not listed as journal
    journal = citation.get("journal", None)
    start_p = citation.get("start", None)
    if (
        journal == "bioRxiv"
        and start_p
        and bib.find("journal=") == -1
        and bib.find("pages=") == -1
    ):
        bpos = bib.rfind("}")
        bib = bib[:bpos].rstrip()
        bib += f", journal={{{journal}}}, pages={{{start_p}}} }}"
        apos = apa.rfind(". http")
        apa = apa[:apos] + f". {journal}, {start_p}" + apa[apos:]
        rpos = ris.rfind("\nER  -")
        ris = ris[:rpos] + f"\nJF  - {journal}\nSP  - {start_p}" + ris[rpos:]
    # Remove DOI from APA for consistency across references
    # apa = re.sub(r'\shttps?://.*$', '', apa)


# Format bibtex
bib = bib.replace("ö", r'{\\"{o}}')
bib = bib.replace("ä", r'{\\"{a}}')
bib = bib.replace("ü", r'{\\"{u}}')
bib = bib.replace("ß", r"{\\ss}")
pos = bib.rfind("}")
if pos != -1:
    bib = bib[:pos] + "\n}\n"
bib = re.sub(r",([^=,]+)\s*=", r",\n    \1=", bib)
bib = re.sub(r"(\s{4,}[\d\w]+)\s*=\s*", r"\1 = ", bib)
(stat_gen / "citation.ris").write_text(ris, "utf-8")
(stat_gen / "citation.bib").write_text(bib, "utf-8")


# -- Package dependencies


def link_pip_package(dep):
    res = re.search(r"([^><=~!]*)([<>=~]*)([^,]*)(.*)", dep)
    assert res is not None, f"Dependency is mal-formatted: {dep}"
    name = res.group(1)
    rel = res.group(2)
    ver = res.group(3)
    ver_full = ver + res.group(4)
    return f"`{name} <https://pypi.org/project/{name}/{ver}>`_{rel}{ver_full}"


py_ver = re.search(r"([<>=~!]*)(.*)", pyproject["project"]["requires-python"])
assert py_ver is not None, "Python-version in pyproject.toml is mal-formatted"
dependencies = (
    "* `python <https://www.python.org/downloads>`_"
    f"{py_ver.group(1)}{py_ver.group(2)}\n"
)
dep_list = pyproject["project"]["dependencies"]
dependencies += "\n".join("* " + link_pip_package(x) for x in dep_list)
(stat_gen / "dependencies.rst").write_text(dependencies, encoding="utf-8")

# -- Substitution

rst_prolog = (
    ".. |citation-apa| replace:: " + apa + "\n"
    ".. |citation-msg| replace:: " + citation_message + "\n"
)
