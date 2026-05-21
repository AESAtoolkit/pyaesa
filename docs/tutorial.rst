Tutorials
=========

The tutorials provide step-by-step examples for setting up a workspace,
preparing input data, selecting study objectives, and running deterministic or
uncertainty workflows.

Use the prerequisite tutorials first, then open the study objective tutorial
that matches the AESA phase and target output: Phase A IO-LCA, Phase B.0
dynamic AR6 carrying capacities, Phase B.1 aSoCC, Phase B.2 aCC, or Phase C
ASR.

Core Prerequisites
------------------

.. toctree::
   :maxdepth: 1

   tutorials/core/0_set_workspace
   tutorials/core/1_download_data
   tutorials/core/2_process_data

Study Objective Guides
----------------------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/0_study_objectives
   tutorials/study_objectives/1_functional_units_and_allocation_methods

Phase A IO-LCA
--------------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/phase_a_iolca_deterministic
   tutorials/study_objectives/phase_a_iolca_uncertainty

Phase B.0 Dynamic AR6 Carrying Capacities
-----------------------------------------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/phase_b0_dynamic_cc_ar6_deterministic
   tutorials/study_objectives/phase_b0_dynamic_cc_ar6_uncertainty

Phase B.1 aSoCC
---------------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/phase_b1_asocc_deterministic
   tutorials/study_objectives/phase_b1_asocc_uncertainty

Phase B.2 aCC
-------------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/phase_b2_acc_deterministic
   tutorials/study_objectives/phase_b2_acc_uncertainty

Phase C ASR
-----------

.. toctree::
   :maxdepth: 1

   tutorials/study_objectives/phase_c_asr_deterministic
   tutorials/study_objectives/phase_c_asr_uncertainty

Optional Workflows
------------------

.. toctree::
   :maxdepth: 1

   tutorials/optional/disaggregate_asocc_mrio_sectors
   tutorials/optional/custom_asocc_method_weights
   tutorials/optional/external_asocc_lca_input_staging

Methodological References
-------------------------

The repository folder ``methodological_notes/`` contains the detailed
methodological PDFs and recommended citation guide. ``set_workspace(...)``
copies these references and the functional unit guide into the active workspace
under ``data_raw/methodological_notes/``. See :doc:`methodological_notes` for
downloadable references.
