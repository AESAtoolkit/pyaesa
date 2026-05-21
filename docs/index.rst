pyaesa Documentation
====================

``pyaesa`` is a Python package for absolute environmental sustainability
assessment (AESA) workflows. It supports data download, data processing,
deterministic calculations, figure rendering, Monte Carlo uncertainty and
Sobol variance.

The documentation is organized for two audiences. The user documentation
explains how to install the package, prepare data, select study objectives,
run deterministic and uncertainty workflows, and cite data and methodological
sources. The developer documentation is separate and is intended for Python
developers modifying package internals.

Start with :doc:`overview` for the package scope and installation. Use
:doc:`workflow_reference` for data sources, prerequisite functions, public
function maps, and study objective routes. Use :doc:`tutorial` for step-by-step
examples and :doc:`api` for the public function reference.

User Documentation
------------------

.. toctree::
   :maxdepth: 2

   overview
   workflow_reference
   tutorial
   api
   methodological_notes

Developer Documentation
-----------------------

The pages below are for Python developers changing package code or adding
allocation methods to the package.

.. toctree::
   :maxdepth: 2

   developer
