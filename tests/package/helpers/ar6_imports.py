"""Short module aliases for AR6 collection and processing package tests."""

from importlib import import_module

_COLLECTION_BASE = "pyaesa.download.ar6"
_PROCESSING_BASE = "pyaesa.process.ar6"

collection_config = import_module(f"{_COLLECTION_BASE}.utils.config")
collection_download = import_module(f"{_COLLECTION_BASE}.download_ar6")
collection_explorer = import_module(f"{_COLLECTION_BASE}.utils.sources.explorer_csv")
collection_historical = import_module(f"{_COLLECTION_BASE}.utils.sources.historical_sources")
collection_iiasa = import_module(f"{_COLLECTION_BASE}.utils.sources.download_iiasa")
collection_public_archive = import_module(f"{_COLLECTION_BASE}.utils.sources.public_archive")
collection_metadata = import_module(f"{_COLLECTION_BASE}.utils.io.metadata")
collection_overlay = import_module(
    f"{_COLLECTION_BASE}.utils.sources.ar6_historical_figure_reference"
)
collection_paths = import_module(f"{_COLLECTION_BASE}.utils.io.paths")
collection_reports = import_module(f"{_COLLECTION_BASE}.utils.io.reports")

processing_contracts = import_module(f"{_PROCESSING_BASE}.utils.io.contracts")
processing_entry = import_module(f"{_PROCESSING_BASE}.process_ar6")
processing_fig_guides = import_module(f"{_PROCESSING_BASE}.utils.figures.figure_guides")
processing_fig_io = import_module(f"{_PROCESSING_BASE}.utils.figures.figure_io")
processing_fig_outputs = import_module(f"{_PROCESSING_BASE}.utils.figures.figure_outputs")
processing_fig_overview = import_module(f"{_PROCESSING_BASE}.utils.figures.figure_overview_panels")
processing_fig_sampling_config = import_module(
    f"{_PROCESSING_BASE}.utils.figures.figure_sampling_config"
)
processing_fig_sampling_panels = import_module(
    f"{_PROCESSING_BASE}.utils.figures.figure_sampling_panels"
)
processing_fig_warming = import_module(f"{_PROCESSING_BASE}.utils.figures.figure_warming_panel")
processing_generate_figures = import_module(f"{_PROCESSING_BASE}.utils.figures.generate_figures")
processing_derived_variables = import_module(f"{_PROCESSING_BASE}.utils.pipeline.derived_variables")
processing_harmonization = import_module(f"{_PROCESSING_BASE}.utils.pipeline.harmonization")
processing_historical = import_module(f"{_PROCESSING_BASE}.utils.pipeline.historical_processing")
processing_loaders = import_module(f"{_PROCESSING_BASE}.utils.pipeline.loaders")
processing_metadata = import_module(f"{_PROCESSING_BASE}.utils.io.metadata")
processing_paths = import_module(f"{_PROCESSING_BASE}.utils.io.paths")
processing_plot_budgets = import_module(f"{_PROCESSING_BASE}.utils.figures.plot_budgets")
processing_plot_helpers = import_module(f"{_PROCESSING_BASE}.utils.figures.plot_helpers")
processing_plot_historical = import_module(f"{_PROCESSING_BASE}.utils.figures.plot_historical")
processing_plot_sampling = import_module(f"{_PROCESSING_BASE}.utils.figures.plot_sampling")
processing_preprocessing = import_module(f"{_PROCESSING_BASE}.utils.pipeline.preprocessing")
processing_process_runner = import_module(f"{_PROCESSING_BASE}.utils.pipeline.process_runner")
processing_processing_modes = import_module(f"{_PROCESSING_BASE}.utils.pipeline.processing_modes")
processing_raw_inputs = import_module(f"{_PROCESSING_BASE}.utils.pipeline.raw_inputs")
processing_report_summaries = import_module(f"{_PROCESSING_BASE}.utils.io.report_summaries")
processing_reports = import_module(f"{_PROCESSING_BASE}.utils.io.reports")
processing_runtime_helpers = import_module(f"{_PROCESSING_BASE}.utils.pipeline.runtime_helpers")
processing_sampling_convergence = import_module(
    f"{_PROCESSING_BASE}.utils.figures.sampling_convergence_utils"
)
processing_sampling_payloads = import_module(f"{_PROCESSING_BASE}.utils.figures.sampling_payloads")
processing_study_period = import_module(f"{_PROCESSING_BASE}.utils.pipeline.study_period")
processing_text_outputs = import_module(f"{_PROCESSING_BASE}.utils.io.text_outputs")
processing_writers = import_module(f"{_PROCESSING_BASE}.utils.io.writers")
