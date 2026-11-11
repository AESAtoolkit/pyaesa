Workspace Guides
================

These pages provide the README guides copied into the workspace by
``set_workspace(...)``, ``prepare_external_inputs(...)``, and
``process_ar6(...)``. Each page identifies the copied file and includes the
package's source guide, so the documentation and workspace instructions use
the same content.

MRIO Aggregation and LCIA
-------------------------

These guides describe custom MRIO classifications, EXIOBASE characterization
matrices, responsibility periods, and LCIA uncertainty inputs. Their files
are copied by ``set_workspace(...)`` under ``data_raw/mrio/``.

.. toctree::
   :maxdepth: 1

   exiobase_aggregation
   oecd_aggregation
   lcia_characterization
   responsibility_periods
   lcia_covs

Carrying Capacities
-------------------

The carrying capacity guide describes custom steady-state carrying capacities
and their matching LCIA methods. Its file is copied by ``set_workspace(...)``
under ``data_raw/carrying_capacities/``.

.. toctree::
   :maxdepth: 1

   carrying_capacities

External LCA and aSoCC Inputs
-----------------------------

These guides specify the external input contracts and examples prepared by
``prepare_external_inputs(...)`` in the project's output folder. See the
:doc:`external input staging tutorial
</tutorials/optional_workflows/external_asocc_lca_input_staging>` for the
guided workflow.

.. toctree::
   :maxdepth: 1

   external_lca
   external_asocc

AR6 Model-Scenario Subsets
--------------------------

The AR6 subset guide accompanies the model-scenario template written by
``process_ar6(...)``. See the :doc:`dynamic AR6 carrying capacity tutorials
</tutorials/study_objectives/(B.0) CC/index>` for the complete workflow.

.. toctree::
   :maxdepth: 1

   ar6_subsets

Methodological Notes and Citations
----------------------------------

The :doc:`methodological notes </methodological_notes>` provide the other
guidance copied by ``set_workspace(...)``: methodological appendices,
recommended citations, and the functional unit and allocation methods guide.
