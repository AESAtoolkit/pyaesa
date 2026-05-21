Package Architecture Notes
==========================

Audience: developers contributing changes to package internals. This section is
not required for normal package use.

These notes explain how the source tree is organized, which module owns which
responsibility, and where the corresponding package tests live.

The stable public Python API is exposed from ``pyaesa/__init__.py`` only.
That package root resolves public exports lazily so subpackage imports do not
pull the entire public API tree into memory during discovery or coverage
startup.
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

- ``pyaesa/ARCHITECTURE_workspace_folders.md``
- ``pyaesa/shared/ARCHITECTURE_shared.md``
- ``pyaesa/shared/figures/ARCHITECTURE_figures.md``
- ``pyaesa/shared/ARCHITECTURE_runtime_scope_and_signature_contracts.md``
- ``pyaesa/shared/uncertainty_assessment/ARCHITECTURE_uncertainty_assessment.md``
- ``pyaesa/workspace_initialisation/ARCHITECTURE_workspace_initialisation.md``
- ``pyaesa/download/ARCHITECTURE_download.md``
- ``pyaesa/process/ARCHITECTURE_process.md``
- ``pyaesa/process/mrios/utils/raw_corrections/ARCHITECTURE_raw_corrections.md``
- ``pyaesa/external_inputs/ARCHITECTURE_external_inputs.md``
- ``pyaesa/asocc/ARCHITECTURE_asocc.md``
- ``pyaesa/acc/ARCHITECTURE_acc.md``
- ``pyaesa/ar6_cc/ARCHITECTURE_ar6_cc.md``
- ``pyaesa/asr/ARCHITECTURE_asr.md``
- ``pyaesa/shared/acc_asr_common/ARCHITECTURE_acc_asr_common.md``
- ``pyaesa/io_lca/ARCHITECTURE_io_lca.md``

The active architecture notes document deterministic owners, uncertainty owners,
and neutral shared helpers. Public uncertainty functions share canonical
request, table, summary, and convergence helpers under
``pyaesa/shared/uncertainty_assessment``.

Method extension checklist:

- ``docs/ADDING_METHODS_checklist.md``


