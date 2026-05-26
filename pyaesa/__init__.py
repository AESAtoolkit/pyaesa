"""
Public functions exported at package level:
- `set_workspace`: Set the active workspace repository and ensure prerequisites
  exist.
- `download_mrio`: Download missing MRIO archives for ``source``.
- `download_pop_gdp`: Download population/GDP datasets used by processing.
- `download_ar6`: Download the raw datasets required for dynamic AR6 climate
  change carrying capacity processing.
- `process_mrio`: Process MRIO archives into ``data_processed`` folder for
  selected years.
- `process_pop_gdp`: Process raw population/GDP data into harmonised analysis
  tables.
- `process_ar6`: Process AR6 scenarios for use by downstream workflows with
  optional harmonization of pathways based on an update of historical
  baselines.
- `deterministic_acc`: Compute deterministic allocated carrying capacities
  (aCC).
- `deterministic_asr`: Compute deterministic absolute sustainability ratio
  (ASR) outputs.
- `deterministic_asocc`: Compute deterministic allocated shares of carrying
  capacities (aSoCC).
- `deterministic_ar6_cc`: Extract AR6 climate change pathways from
  ``process_ar6(...)`` outputs.
- `disaggregate_asocc`: Disaggregate non LCIA deterministic allocated shares
  of carrying capacities (aSoCC).
- `deterministic_io_lca`: Compute deterministic IO-LCA outputs from processed
  MRIO tables.
- `prepare_external_inputs`: Import external allocated shares of carrying
  capacities (aSoCC) and LCA inputs.
- `write_asocc_weight_template`: Write equal weight inter-method templates for
  one method scope.
- `preview_asocc_weight_tree`: Validate an edited custom inter-method tree and
  render its preview.
- `uncertainty_asocc`: Run allocated shares of carrying capacities (aSoCC)
  Monte Carlo uncertainty.
- `uncertainty_io_lca`: Run IO-LCA Monte Carlo uncertainty from deterministic
  IO-LCA outputs.
- `uncertainty_ar6_cc`: Run dynamic AR6 carrying capacity (CC) Monte Carlo
  uncertainty.
- `uncertainty_acc`: Run allocated carrying capacity (aCC) Monte Carlo
  uncertainty.
- `uncertainty_asr`: Run absolute sustainability ratio (ASR) Monte Carlo
  uncertainty.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.2.1"

_PUBLIC_EXPORT_OWNERS = {
    "set_workspace": "pyaesa.workspace_initialisation.set_workspace",
    "download_mrio": "pyaesa.download.mrios.download_mrio",
    "download_pop_gdp": "pyaesa.download.pop_gdp.download_pop_gdp",
    "download_ar6": "pyaesa.download.ar6.download_ar6",
    "process_mrio": "pyaesa.process.mrios.process_mrio",
    "process_pop_gdp": "pyaesa.process.pop_gdp.process_pop_gdp",
    "process_ar6": "pyaesa.process.ar6.process_ar6",
    "deterministic_acc": "pyaesa.acc.deterministic_acc",
    "deterministic_asr": "pyaesa.asr.deterministic_asr",
    "deterministic_asocc": "pyaesa.asocc.deterministic_asocc",
    "deterministic_ar6_cc": "pyaesa.ar6_cc.deterministic_ar6_cc",
    "disaggregate_asocc": "pyaesa.asocc.disaggregate_asocc",
    "deterministic_io_lca": "pyaesa.io_lca.deterministic_io_lca",
    "prepare_external_inputs": "pyaesa.external_inputs.prepare_external_inputs",
    "write_asocc_weight_template": "pyaesa.asocc.inter_method_weights",
    "preview_asocc_weight_tree": "pyaesa.asocc.inter_method_weights",
    "uncertainty_asocc": "pyaesa.asocc.uncertainty_asocc",
    "uncertainty_io_lca": "pyaesa.io_lca.uncertainty_io_lca",
    "uncertainty_ar6_cc": "pyaesa.ar6_cc.uncertainty_ar6_cc",
    "uncertainty_acc": "pyaesa.acc.uncertainty_acc",
    "uncertainty_asr": "pyaesa.asr.uncertainty_asr",
}

if TYPE_CHECKING:
    from .acc.deterministic_acc import deterministic_acc
    from .acc.uncertainty_acc import uncertainty_acc
    from .ar6_cc.deterministic_ar6_cc import deterministic_ar6_cc
    from .ar6_cc.uncertainty_ar6_cc import uncertainty_ar6_cc
    from .asocc.deterministic_asocc import deterministic_asocc
    from .asocc.disaggregate_asocc import disaggregate_asocc
    from .asocc.inter_method_weights import preview_asocc_weight_tree, write_asocc_weight_template
    from .asocc.uncertainty_asocc import uncertainty_asocc
    from .asr.deterministic_asr import deterministic_asr
    from .asr.uncertainty_asr import uncertainty_asr
    from .download.ar6.download_ar6 import download_ar6
    from .download.mrios.download_mrio import download_mrio
    from .download.pop_gdp.download_pop_gdp import download_pop_gdp
    from .external_inputs.prepare_external_inputs import prepare_external_inputs
    from .io_lca.deterministic_io_lca import deterministic_io_lca
    from .io_lca.uncertainty_io_lca import uncertainty_io_lca
    from .process.ar6.process_ar6 import process_ar6
    from .process.mrios.process_mrio import process_mrio
    from .process.pop_gdp.process_pop_gdp import process_pop_gdp
    from .workspace_initialisation.set_workspace import set_workspace

__all__ = [
    "set_workspace",
    "download_mrio",
    "download_pop_gdp",
    "download_ar6",
    "process_mrio",
    "process_pop_gdp",
    "process_ar6",
    "deterministic_acc",
    "deterministic_asr",
    "deterministic_asocc",
    "deterministic_ar6_cc",
    "disaggregate_asocc",
    "deterministic_io_lca",
    "prepare_external_inputs",
    "write_asocc_weight_template",
    "preview_asocc_weight_tree",
    "uncertainty_asocc",
    "uncertainty_io_lca",
    "uncertainty_ar6_cc",
    "uncertainty_acc",
    "uncertainty_asr",
]


def __getattr__(name):
    """Load package-level public functions only when first accessed."""
    if name not in _PUBLIC_EXPORT_OWNERS:
        raise AttributeError(f"module 'pyaesa' has no attribute {name!r}")

    module = import_module(_PUBLIC_EXPORT_OWNERS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Expose the package-level public API for interactive discovery."""
    return sorted(set(globals()) | set(__all__))
