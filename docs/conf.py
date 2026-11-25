# Configuration file for the Sphinx documentation builder.
# http://www.sphinx-doc.org/en/master/config

import os
import re
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'eddy'
copyright = '2020-2026, Richard Teague'
author = 'Richard Teague'
# Read the version from the package rather than hardcoding it here, so the
# docs cannot drift from eddy/__init__.py. Parsed, not imported, because the
# package's runtime dependencies are not installed on the docs builder.
def _get_version():
    init = os.path.join(os.path.dirname(__file__), '..', 'eddy', '__init__.py')
    with open(init) as f:
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f.read(), re.M)
    return match.group(1) if match else '0.0.0'


release = _get_version()
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    'sphinx_copybutton',
    'nbsphinx',
]

# Math is rendered at build time by KaTeX (Computer Modern), so equations
# read as maths rather than as code and no JS runs in the reader's browser.
katex_prerender = True

# sphinxcontrib-katex defaults to loading katex.min.css from jsdelivr. The
# brand calls for no third-party network dependency, so the stylesheet and
# its woff2 faces are vendored under _static/katex/ and served locally.
katex_css_path = 'katex/katex.min.css'

autodoc_mock_imports = ['astropy']

# nbsphinx loads RequireJS from cdnjs to support ipywidgets. None of the
# tutorials use widgets, so switch it off rather than keep the CDN call --
# same reason the fonts and KaTeX CSS are self-hosted.
nbsphinx_requirejs_path = ''

templates_path = ['_templates']
master_doc = 'index'
# The structure function tutorial is being revised; a coming-soon page stands
# in for it in the toctree until it is ready to publish.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    'tutorials/tutorial_7_structurefunction.ipynb']
add_function_parentheses = True

# -- HTML output — Planet Formation Lab brand --------------------------------

html_theme = 'furo'
html_title = 'eddy'
html_static_path = ['_static']
html_css_files = ['css/eddy.css']
html_js_files = ['js/eddy-deprojection.js']
html_favicon = '_static/favicon.png'
html_show_sourcelink = False

# Pygments: a custom style in the PFL palette (docs/_pygments/pfl.py),
# importable because this directory is on sys.path above.
pygments_style = '_pygments.pfl.PFLStyle'
# Furo still emits a dark code style for OS dark mode even though the rest
# of the page stays light, so point it at the same style.
pygments_dark_style = '_pygments.pfl.PFLStyle'

# Light ("paper") only. Furo always ships a toggle, so the dark
# variables are set to the light values and the toggle is hidden in
# eddy.css — the docs render one way regardless of OS preference.
_PFL = {
    'color-brand-primary': '#C24E00',
    'color-brand-content': '#136E7A',
    'color-background-primary': '#FFFFFF',
    'color-background-secondary': '#FBFAF7',
    'color-foreground-primary': '#1A1916',
    'color-foreground-secondary': '#585548',
    'color-sidebar-background': '#FFFFFF',
}

# No search box: furo's sidebar/search.html is left out of the sidebar.
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar/ethical-ads.html',
        'sidebar/scroll-end.html',
        'sidebar/variant-selector.html',
    ]
}

html_theme_options = {
    'light_logo': 'eddy_mark_v2.png',
    'dark_logo': 'eddy_mark_v2.png',
    'sidebar_hide_name': False,
    'light_css_variables': _PFL,
    'dark_css_variables': _PFL,
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/PlanetFormationLab/eddy',
            'html': '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.06-.15-.36-.76.07-1.58 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.43.82.13 1.43.07 1.58.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A7.995 7.995 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path></svg>',  # noqa: E501 (inline SVG icon)
            'class': '',
        },
        {
            'name': 'Planet Formation Lab',
            'url': 'https://pfl.mit.edu',
            'html': '<svg stroke="currentColor" fill="none" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.2"></circle><ellipse cx="12" cy="12" rx="10" ry="4.2" transform="rotate(-18 12 12)"></ellipse></svg>',  # noqa: E501 (inline SVG icon)
            'class': '',
        },
    ],
}
