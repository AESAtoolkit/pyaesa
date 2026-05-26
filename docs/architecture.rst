Package Architecture Notes
==========================

Audience: developers contributing changes to package internals. This section is
not required for normal package use.

These notes explain how the source tree is organized, which module owns which
responsibility, and where the corresponding package tests live.

The stable public Python API is exposed from
`pyaesa/__init__.py <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/__init__.py>`__
only. That package root resolves public exports lazily so subpackage imports
do not pull the entire public API tree into memory during discovery or
coverage startup.
Non root ``pyaesa/**/__init__.py`` files are package markers rather than
secondary public facades. Internal imports should target concrete owner
modules rather than package re export layers.

Each architecture note documents the same core information where relevant:

- purpose
- public surface
- responsibility boundary
- internal organization and key runtime contracts
- testing and quality gates

Architecture notes live alongside the package source tree:

- `pyaesa/ARCHITECTURE_workspace_folders.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/ARCHITECTURE_workspace_folders.md>`__
- `pyaesa/shared/ARCHITECTURE_shared.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/shared/ARCHITECTURE_shared.md>`__
- `pyaesa/shared/figures/ARCHITECTURE_figures.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/shared/figures/ARCHITECTURE_figures.md>`__
- `pyaesa/shared/ARCHITECTURE_runtime_scope_and_signature_contracts.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/shared/ARCHITECTURE_runtime_scope_and_signature_contracts.md>`__
- `pyaesa/shared/uncertainty_assessment/ARCHITECTURE_uncertainty_assessment.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/shared/uncertainty_assessment/ARCHITECTURE_uncertainty_assessment.md>`__
- `pyaesa/workspace_initialisation/ARCHITECTURE_workspace_initialisation.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/workspace_initialisation/ARCHITECTURE_workspace_initialisation.md>`__
- `pyaesa/download/ARCHITECTURE_download.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/download/ARCHITECTURE_download.md>`__
- `pyaesa/process/ARCHITECTURE_process.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/process/ARCHITECTURE_process.md>`__
- `pyaesa/process/mrios/utils/raw_corrections/ARCHITECTURE_raw_corrections.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/process/mrios/utils/raw_corrections/ARCHITECTURE_raw_corrections.md>`__
- `pyaesa/external_inputs/ARCHITECTURE_external_inputs.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/external_inputs/ARCHITECTURE_external_inputs.md>`__
- `pyaesa/asocc/ARCHITECTURE_asocc.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asocc/ARCHITECTURE_asocc.md>`__
- `pyaesa/acc/ARCHITECTURE_acc.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/acc/ARCHITECTURE_acc.md>`__
- `pyaesa/ar6_cc/ARCHITECTURE_ar6_cc.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/ar6_cc/ARCHITECTURE_ar6_cc.md>`__
- `pyaesa/asr/ARCHITECTURE_asr.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/asr/ARCHITECTURE_asr.md>`__
- `pyaesa/shared/acc_asr_common/ARCHITECTURE_acc_asr_common.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/shared/acc_asr_common/ARCHITECTURE_acc_asr_common.md>`__
- `pyaesa/io_lca/ARCHITECTURE_io_lca.md <https://github.com/AESAtoolkit/pyaesa/blob/main/pyaesa/io_lca/ARCHITECTURE_io_lca.md>`__

The active architecture notes document deterministic owners, uncertainty owners,
and neutral shared helpers. Public uncertainty functions share canonical
request, table, summary, and convergence helpers under
``pyaesa/shared/uncertainty_assessment``.

Method extension checklist:

- :doc:`docs/ADDING_METHODS_checklist.md <ADDING_METHODS_checklist>`

