# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


### SETTING ROOT DIR

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../blueprints'))
sys.path.insert(0, os.path.abspath('/home/mortimer/home/Projects/blueprints'))

sys.path.insert(0, str(Path('../..', 'blueprints').resolve()))

###


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'blueprints'
copyright = '2026, Mortimer von Chappuis'
author = 'Mortimer von Chappuis'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 
	      'sphinx.ext.viewcode', 
	      'sphinx.ext.autodoc', 
	      'sphinx.ext.inheritance_diagram', 
	      'sphinx.ext.autosummary', 
	      'sphinx.ext.mathjax', 
	      'sphinx_automodapi.automodapi',
	      'sphinx.ext.napoleon', 
	      'sphinx.ext.graphviz', 
	      'sphinx.ext.intersphinx', 
	      #'sphinxcontrib.youtube', 
              #'sphinx_automodapi.smart_resolver',
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
#html_theme = 'renku'
#html_theme = 'python_docs_theme'
html_theme = 'pydata_sphinx_theme'
#html_theme = 'furo'
#html_theme = "sphinx_book_theme"


html_context = {
   "default_mode": "auto"
}
html_theme_options = {
   "pygment_light_style": "tango",
   "pygment_dark_style": "monokai", 
   'navbar_start': ['navbar-logo'],
   "header_links_before_dropdown": 10,
   "show_toc_level": 1,
   "external_links": [
        #{
        #    "url": "https://github.com/mortimervonchappuis/mujoco_blueprints",
        #    "name": "Github",
        #},
        {
            "url": "https://mujoco.readthedocs.io/en/stable/overview.html",
            "name": "Mujoco",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mortimervonchappuis/mujoco_blueprints",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/mujoco-blueprints/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "PyData",
            "url": "https://pydata.org",
            "icon": "fa-custom fa-pydata",
        },
    ],
    "logo": {
        "text": "Mujoco Blueprints",
        "image_dark": "_static/logo_margin_thick.png",
    },

   #'footer_end': [],
   # OWN MODIFICATIONS
}

inheritance_graph_attrs = dict(rankdir="LR", size='"100.0, 40.0"',
                               fontsize=24, ratio='compress')
inheritance_node_attrs = dict(shape='box', fontsize=42, height=2.,
                              color='darkorange', style='filled')



html_sidebars = {
    #"**": ["sidebar-nav-bs"], 
    "blueprints/*": ['sidebar-nav-bs', 'page-toc']
}



autodoc_class_signature = "separated"
autodoc_typehints = "description"
autodoc_member_order = 'bysource'
autodoc_default_options = {'inherited-members': False,#True, 
			   'member-order': 'bysource',
			   'undoc-members': True,
			   'private-members': False, 

			   #'show-inheritance':  True,
			   #'exclude-members': ['unique']
			   }
#exclude_patterns = ["**/*Type.rst",]






napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False#True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True


html_static_path = ['_static']
html_logo = "_static/logo_margin_thick.png"