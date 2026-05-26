import json
from pathlib import Path

from pyaesa.workspace_initialisation.workspace import clear_default_repo_root, set_default_repo_root
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_A_LCA,
    PHASE_B2_ACC,
    phase_label_for_owner,
    public_phase_reuse_status,
    public_phase_index_reuse_status,
    phase_index_path_for_metadata,
    phase_ready_detail,
    phase_reused_detail,
    read_phase_index,
    write_phase_index,
)
from pyaesa.shared.runtime.reporting.labels import plural_label
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.progress import StatusProgressPrinter, YearProgressPrinter
from pyaesa.shared.runtime.reporting.run_progress import (
    monte_carlo_completion_is_persistent,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress_label,
    monte_carlo_run_progress,
    sobol_progress,
    visible_status_for_run_work,
)
from pyaesa.shared.runtime.reporting.summary import document, render_summary, section
from pyaesa.shared.runtime.reporting.values import (
    as_sequence,
    format_report_value,
    format_ssp_value,
    format_summary_value,
    format_values,
)
from pyaesa.shared.runtime.reporting.year_ranges import format_year_ranges
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest, write_manifest
from pyaesa.shared.uncertainty_assessment.orchestration import (
    complete_uncertainty_manifest_phase,
    progress_complete,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_dependencies import dependency_section
from pyaesa.shared.uncertainty_assessment.run_state.report import uncertainty_report


def _uncertainty_scope_artifacts(tmp_path: Path, run_id: str) -> dict[str, str]:
    return {"scope_manifest": str(tmp_path / run_id / "logs" / "scope_manifest.json")}


def _write_io_lca_dependency_manifest(
    path: Path,
    *,
    include_source: bool = True,
    include_tables: bool = True,
    include_figures: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = path.parent / "figures" / "diagnostic.png"
    arguments = {
        "agg_reg": False,
        "agg_sec": True,
        "agg_version": "ixi",
        "group_indices": False,
    }
    if include_source:
        arguments["source"] = "exiobase_3"
    sections: dict[str, object] = {}
    if include_tables:
        sections.update(
            {
                "main": {"rows": 1},
                "origin": {"rows": 1},
                "stages": {"rows": 1},
            }
        )
    if include_figures:
        sections["figures"] = {"diagnostic": {"paths": [str(figure_path)]}}
    path.write_text(
        json.dumps(
            {
                "arguments": arguments,
                "artifacts": {"paths_written": [str(path.parent / "results" / "main.csv")]},
                "execution": {"sections": sections},
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "summary.log").write_text("summary\n", encoding="utf-8")


def _write_uncertainty_dependency_manifest(path: Path, *, family: str) -> None:
    manifest = build_manifest(
        family=family,
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=1,
        status="complete",
        run_id=f"mc_nested_{family}",
        artifacts={"scope_manifest": str(path)},
    )
    write_manifest(path=path, manifest=manifest)


def _process_ar6_payload(output_root: Path) -> dict[str, object]:
    return {
        "reuse_status": "computed",
        "study_period": "2020-2030",
        "categories": ["C1"],
        "ssps": ["SSP2"],
        "harmonization": True,
        "harmonization_method": "offset",
        "output_root": str(output_root),
        "output_files_available": 1,
        "figures_available": 0,
        "variable_coverage": [],
    }


class _RecordingStatus:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.persistent: list[bool] = []
        self.finished = False
        self.cleared = False

    def show(self, message: str) -> None:
        self.messages.append(str(message))
        self.persistent.append(False)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        self.messages.append(str(message))
        self.persistent.append(bool(persistent))

    def finish(self) -> None:
        self.finished = True

    def clear_transient(self) -> None:
        self.cleared = True


def test_uncertainty_report_writes_summary_log(tmp_path: Path) -> None:
    deterministic_asocc_manifest = tmp_path / "det_asocc" / "logs" / "scope_manifest.json"
    deterministic_asocc_manifest.parent.mkdir(parents=True)
    deterministic_asocc_manifest.write_text(
        json.dumps(
            {
                "arguments": {
                    "source": "exiobase_3",
                    "agg_reg": True,
                    "agg_sec": False,
                    "agg_version": "ixi",
                    "group_indices": False,
                    "r_p": ["FR"],
                    "s_p": ["D"],
                    "r_c": None,
                    "r_f": None,
                }
            }
        ),
        encoding="utf-8",
    )
    io_lca_manifest = tmp_path / "io_lca" / "logs" / "scope_manifest.json"
    _write_io_lca_dependency_manifest(io_lca_manifest)
    io_lca_minimal_manifest = tmp_path / "io_lca_minimal" / "logs" / "scope_manifest.json"
    _write_io_lca_dependency_manifest(
        io_lca_minimal_manifest,
        include_source=False,
        include_tables=False,
        include_figures=False,
    )
    outputs = {
        "scope_manifest": str(
            tmp_path / "report_demo" / "monte_carlo" / "run_1" / "logs" / "scope.json"
        ),
        "public_row_identity": str(tmp_path / "public_row_identity.csv"),
        "run_values": str(tmp_path / "asocc_runs.csv"),
        "asocc_runs": str(tmp_path / "asocc_runs.csv"),
        "summary_stats_runs": str(tmp_path / "summary.csv"),
        "results_readme": str(tmp_path / "README.txt"),
        "source_methods": str(tmp_path / "source_methods.csv"),
        "figure_paths": [str(tmp_path / "figures" / "figure.png")],
        "sobol_indices": str(tmp_path / "sobol" / "sobol_indices.csv"),
        "sobol_source_summary": str(tmp_path / "sobol" / "sobol_summary.csv"),
        "sobol_readme": str(tmp_path / "sobol" / "README_sobol.txt"),
    }
    manifest = build_manifest(
        family="asocc",
        mode="convergence",
        output_format="csv_compact",
        active_sources=("projection_uncertainty", "reference_year_uncertainty"),
        completed_runs=500,
        requested_runs=500,
        status="complete",
        run_id="mc_report_demo",
        arguments={
            "project_name": "report_demo",
            "years": [2030],
            "lcia_method": ["gwp100_lcia"],
            "fu_code": "L2.a.a",
            "source": "exiobase_3",
            "agg_reg": True,
            "agg_sec": False,
            "agg_version": "ixi",
            "group_indices": False,
            "version_name": "v1",
            "r_p": ["FR"],
            "s_p": ["D"],
        },
        source_parameters={"inter_mrio_uncertainty": {"source": "oecd_v2025"}},
        artifacts=outputs,
        deterministic_prerequisites=(
            {
                "base_function_source": "deterministic_asocc",
                "scope_manifest": str(deterministic_asocc_manifest),
                "reuse_status": "computed",
            },
            {
                "base_function_source": "deterministic_ar6_cc",
                "scope_key": "cc_scope",
                "categories": ["C1"],
                "ssp_scenarios": ["SSP2"],
                "deterministic_paths": [str(tmp_path / "ar6_cc" / "results.csv"), " "],
                "figure_paths": [str(tmp_path / "ar6_cc" / "figure.png")],
                "process_ar6": _process_ar6_payload(tmp_path / "process_ar6_write"),
            },
            {
                "base_function_source": "deterministic_io_lca",
                "metadata_path": str(io_lca_manifest),
                "reuse_status": "computed",
            },
            {
                "base_function_source": "io_lca_deterministic",
                "metadata_path": str(io_lca_minimal_manifest),
                "reuse_status": "computed",
            },
        ),
        external_inputs=(
            {
                "type": "external_lca_monte_carlo",
                "version_name": "supplier",
                "reuse_status": "computed",
                "output_root": str(tmp_path / "external_lca"),
                "figures_available": 1,
            },
            {
                "type": "external_lca_deterministic",
                "version_name": "supplier_fixed",
                "reuse_status": "computed",
                "output_root": str(tmp_path / "external_lca_fixed"),
                "output_file": str(tmp_path / "external_lca_fixed" / "results.csv"),
                "post_study_output_file": str(tmp_path / "external_lca_fixed" / "post_study.csv"),
                "deterministic_paths": [
                    str(tmp_path / "external_lca_fixed" / "deterministic.csv"),
                    " ",
                ],
                "figures_available": 0,
            },
            {
                "selection": "CO(S)::UT(FD)",
                "storage_mode": "deterministic",
            },
        ),
        convergence={"reached": False, "max_runs": 500},
        sobol={"ran": True, "reached": False, "n_base_samples": 128},
    )

    report = uncertainty_report(manifest=manifest, reuse_status="computed")
    text = str(report)

    assert report.manifest is manifest
    assert report.manifest.run_id == "mc_report_demo"
    assert text
    assert all(value in text for value in ("report_demo", "exiobase_3", "ixi"))
    assert "oecd_v2025" in text
    summary_log = Path(outputs["scope_manifest"]).parent / "summary.log"
    assert summary_log.read_text(encoding="utf-8").strip() == text
    phase = PhasePrinter("uncertainty_asocc")
    complete_uncertainty_manifest_phase(
        phase=phase,
        scope_name="aSoCC uncertainty",
        report=uncertainty_report(manifest=manifest, reuse_status="reused_exact"),
    )
    phase.finish()


def test_uncertainty_report_covers_empty_and_sobol_status_variants(tmp_path: Path) -> None:
    empty_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=0,
        status="complete",
        run_id="mc_empty",
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_empty"),
    )
    assert str(uncertainty_report(manifest=empty_manifest, reuse_status="reused_exact"))
    no_owner_arguments = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=0,
        status="complete",
        run_id="mc_no_owner_arguments",
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_no_owner_arguments"),
    )
    assert str(uncertainty_report(manifest=no_owner_arguments, reuse_status="computed"))
    no_alternate_source = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=0,
        status="complete",
        run_id="mc_no_alternate_source",
        arguments={"agg_reg": False},
        source_parameters={"inter_mrio_uncertainty": {}},
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_no_alternate_source"),
    )
    assert str(uncertainty_report(manifest=no_alternate_source, reuse_status="computed"))

    skipped_sobol = build_manifest(
        family="io_lca",
        mode="fixed",
        output_format="csv_compact",
        arguments={
            "base_io_lca_args": {
                "project_name": "io_lca_report_demo",
                "source": "exiobase_3",
                "agg_reg": False,
                "agg_sec": True,
                "agg_version": "ixi",
                "years": [2020],
                "lcia_method": ["gwp100_lcia"],
                "fu_code": "L2.a.a",
                "r_f": ["FR"],
                "r_c": None,
                "r_p": None,
                "s_p": ["D"],
                "group_indices": False,
            }
        },
        active_sources=("lcia_uncertainty",),
        completed_runs=2,
        status="complete",
        run_id="mc_sobol_skip",
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_sobol_skip"),
        sobol={"ran": False, "reason": "not enough sources"},
    )
    skipped_text = str(uncertainty_report(manifest=skipped_sobol, reuse_status="computed"))
    assert skipped_text
    assert all(value in skipped_text for value in ("io_lca_report_demo", "exiobase_3", "ixi"))

    ran_sobol = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("a", "b"),
        completed_runs=2,
        status="complete",
        run_id="mc_sobol_ran",
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_sobol_ran"),
        sobol={"ran": True},
    )
    assert str(uncertainty_report(manifest=ran_sobol, reuse_status="computed"))

    reached_sobol = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("a", "b"),
        completed_runs=2,
        status="complete",
        run_id="mc_sobol_reached",
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_sobol_reached"),
        convergence={"reached": True},
        sobol={"ran": True, "reached": True},
    )
    reached_text = str(uncertainty_report(manifest=reached_sobol, reuse_status="computed"))
    assert reached_text

    ar6_category_manifest = build_manifest(
        family="ar6_cc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("dynamic_ar6_cc_uncertainty",),
        completed_runs=2,
        status="complete",
        run_id="mc_ar6_category",
        arguments={
            "base_ar6_cc_args": {
                "project_name": "ar6_report_demo",
                "years": [2030],
                "ssp_scenarios": ["SSP2"],
            }
        },
        deterministic_prerequisites=(
            {
                "base_function_source": "deterministic_ar6_cc",
                "categories": ["C1", "C2"],
                "ssp_scenarios": ["SSP2", "SSP3"],
                "emissions_mode": "gross",
                "process_ar6": _process_ar6_payload(tmp_path / "process_ar6_category"),
                "pathway_counts": [
                    {
                        "category": "C1",
                        "ssp_scenario": "SSP2",
                        "model_scenario_pairs": 2,
                    }
                ],
            },
        ),
        external_inputs=(
            {"source": "external_lca"},
            {
                "selection": "supplier",
                "storage_mode": "monte_carlo",
            },
        ),
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_ar6_category"),
        source_parameters={"dynamic_ar6_cc_uncertainty": {"category_uncertainty": True}},
        lineage={"source_inventory": {"scope_availability_messages": ["AR6 source warning"]}},
    )
    ar6_category_text = str(
        uncertainty_report(manifest=ar6_category_manifest, reuse_status="computed")
    )
    assert "AR6 source warning" in ar6_category_text

    io_lca_scope_manifest = (
        tmp_path / "io_lca_dependency" / "deterministic" / "logs" / "scope_manifest.json"
    )
    _write_io_lca_dependency_manifest(io_lca_scope_manifest)
    convergence_manifest = build_manifest(
        family="asr",
        mode="convergence",
        output_format="csv_compact",
        arguments={
            "project_name": "demo",
            "years": [2020, 2021, 2030],
            "lcia_method": "gwp100_lcia",
            "fu_code": "L2.a.a",
        },
        active_sources=("external_lca::supplier",),
        completed_runs=4,
        requested_runs=6,
        status="complete",
        run_id="mc_convergence_reason",
        convergence={"reached": False, "max_runs": 6, "completed_runs": 4, "reason": "stable"},
        sobol={
            "ran": True,
            "mode": "convergence",
            "reached": False,
            "n_base_samples": 8,
            "max_base_samples": 16,
            "active_source_count": 2,
        },
        lineage={
            "summary_records": [
                "ignored",
                {"severity": "INFO", "message": "not a warning"},
                {"severity": "WARNING", "message": ""},
                {"severity": "WARNING", "message": "lineage warning"},
            ]
        },
        deterministic_prerequisites=(
            {
                "base_function_source": "deterministic_ar6_cc",
                "process_ar6": {
                    "reuse_status": "computed",
                    "study_period": [2020, 2021, 2030],
                    "categories": ["C1"],
                    "ssps": ["SSP2"],
                    "harmonization": "offset",
                    "figures_available": 1,
                    "variable_coverage": [
                        {
                            "variable": "Emissions|CO2",
                            "retained_model_scenario_pairs": 1,
                            "available_model_scenario_pairs": 2,
                        }
                    ],
                },
                "missing_pathway_combinations": [
                    {"category": "C9", "ssp_scenario": "SSP9", "model_scenario_pairs": 0}
                ],
                "metadata_path": str(
                    Path("demo") / "B2_acc" / "deterministic" / "logs" / "scope_manifest.json"
                ),
            },
        ),
        external_inputs=(
            {
                "source": "external_lca_monte_carlo",
                "reuse_status": "reused_exact",
                "version_name": "supplier",
                "lcia_method": "gwp100_lcia",
                "output_root": str(Path("demo") / "A_lca"),
                "figures_available": 2,
                "summary_records": [
                    {"severity": "INFO", "message": {"message": "input info"}},
                    {"severity": "WARNING", "message": ["input warning", "input warning"]},
                ],
            },
            {
                "source": "io_lca_deterministic",
                "scope_manifest": str(io_lca_scope_manifest),
            },
        ),
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_convergence_reason"),
        compatibility_context={"artifact_contract": "asr_branch_set"},
    )
    convergence_text = str(
        uncertainty_report(manifest=convergence_manifest, reuse_status="computed")
    )
    assert convergence_text
    assert "Convergence scope: independent per branch" in convergence_text
    assert "lineage warning" in convergence_text

    nested_acc_manifest = tmp_path / "nested_acc" / "monte_carlo" / "mc_1" / "logs" / "scope.json"
    _write_uncertainty_dependency_manifest(nested_acc_manifest, family="acc")
    edge_manifest = build_manifest(
        family="ar6_cc",
        mode="convergence",
        output_format="csv_compact",
        arguments={
            "base_ar6_cc_args": {
                "project_name": "edge_ar6_report",
                "years": [2020],
                "ssp_scenarios": ["SSP2"],
            }
        },
        active_sources=(),
        completed_runs=1,
        requested_runs=3,
        status="complete",
        run_id="mc_report_edges",
        sobol={"ran": False},
        lineage={"source_inventory": ["not a mapping"]},
        deterministic_prerequisites=(
            {
                "source": "deterministic_acc",
                "scope_manifest": str(
                    Path("demo") / "B2_acc" / "deterministic" / "logs" / "scope_manifest.json"
                ),
            },
            {
                "base_function_source": "deterministic_ar6_cc",
                "process_ar6": {
                    "reuse_status": "computed",
                    "variable_coverage": [
                        "ignored",
                        {
                            "variable": "Emissions|CO2",
                            "retained_model_scenario_pairs": 1,
                            "available_model_scenario_pairs": 2,
                        },
                    ],
                    "figures_available": 0,
                },
            },
            {
                "base_function_source": "deterministic_ar6_cc",
                "process_ar6": {
                    "reuse_status": "computed",
                    "variable_coverage": [],
                    "figures_available": 0,
                },
            },
        ),
        external_inputs=(
            {
                "source": "uncertainty_acc",
                "scope_manifest": str(nested_acc_manifest),
                "summary_arguments": {},
                "output_file": str(tmp_path / "nested_acc" / "results.csv"),
                "post_study_output_file": str(tmp_path / "nested_acc" / "post_study.csv"),
                "deterministic_paths": [str(tmp_path / "nested_acc" / "deterministic.csv")],
            },
            {
                "source": "external_lca_monte_carlo",
                "figures_available": 0,
                "warning_messages": 12,
                "summary_records": [
                    2,
                    {"severity": "WARNING", "message": 34},
                ],
            },
        ),
        artifacts=_uncertainty_scope_artifacts(tmp_path, "mc_report_edges"),
    )
    edge_text = str(uncertainty_report(manifest=edge_manifest, reuse_status="computed"))
    assert edge_text


def test_dependency_section_covers_summary_argument_and_inventory_edges(tmp_path: Path) -> None:
    io_lca_scope_manifest = tmp_path / "io_lca" / "deterministic" / "logs" / "scope.json"
    _write_io_lca_dependency_manifest(
        io_lca_scope_manifest,
        include_source=False,
        include_tables=False,
        include_figures=False,
    )

    no_scope_arguments = dependency_section(
        payload={
            "source": "io_lca_deterministic",
            "scope_manifest": str(io_lca_scope_manifest),
            "summary_arguments": {},
        }
    )
    assert no_scope_arguments.lines

    nested_dependency = dependency_section(
        payload={
            "source": "uncertainty_acc",
            "summary_arguments": {"source": "exiobase_3"},
            "output_file": str(tmp_path / "nested" / "results.csv"),
            "post_study_output_file": str(tmp_path / "nested" / "post_study.csv"),
            "deterministic_paths": [str(tmp_path / "nested" / "deterministic.csv"), " "],
            "scope_manifest": str(tmp_path / "nested" / "logs" / "scope.json"),
        }
    )

    assert nested_dependency.lines


def test_run_progress_factories_emit_replaceable_lines(capsys) -> None:
    monte_carlo = monte_carlo_run_progress(source="uncertainty_test")
    monte_carlo.begin(label=monte_carlo_run_drawing_label(start=0, stop=5, max_runs=10))
    monte_carlo.complete(label=monte_carlo_run_progress_label(completed=5, max_runs=10))
    monte_carlo.skip()
    monte_carlo.finish()

    sobol = sobol_progress(source="uncertainty_test")
    sobol_label = "base samples 128 (max 1024); design evaluations 512"
    sobol.begin(label=sobol_label)
    sobol.complete(label=sobol_label)
    sobol.finish()

    captured = capsys.readouterr()
    assert captured.out
    assert "uncertainty_test" in captured.out

    long_progress = sobol_progress(source="uncertainty_test")
    long_progress.begin(label="base samples " + "1234567890" * 12)
    long_progress.finish()
    long_output = capsys.readouterr().out.split("\r")[-1].strip()
    assert long_output.endswith("...")
    assert len(long_output) <= 100


def test_run_progress_disabled_mode_is_silent(capsys) -> None:
    progress = monte_carlo_run_progress(source="uncertainty_test", enabled=False)
    progress.begin(label=monte_carlo_run_drawing_label(start=0, stop=10, max_runs=10))
    progress.complete(label=monte_carlo_run_progress_label(completed=10, max_runs=10))
    progress.show("hidden")
    progress.skip()
    progress.clear_transient()
    progress.finish()

    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_progress_uses_status_sink() -> None:
    status = _RecordingStatus()
    progress = monte_carlo_run_progress(source="uncertainty_test", status=status)

    progress.begin(label=monte_carlo_run_drawing_label(start=0, stop=1, max_runs=2))
    progress.complete(label=monte_carlo_run_progress_label(completed=1, max_runs=2))
    progress.show("transient")
    progress.finish()

    assert len(status.messages) >= 3
    assert status.messages[-1] == "transient"
    assert status.finished is False
    fixed_label = monte_carlo_run_progress_label(completed=2, max_runs=2, mode="fixed")
    assert fixed_label
    assert "2" in fixed_label
    assert (
        monte_carlo_run_drawing_label(start=0, stop=2, max_runs=2, mode="fixed")
        == "drawing fixed runs 1 to 2 of 2"
    )
    assert monte_carlo_run_drawing_label(
        start=2,
        stop=4,
        max_runs=10,
        component=True,
    ).endswith("component checkpoint")

    terminal_progress = monte_carlo_run_progress(source="uncertainty_test")
    terminal_progress.show("terminal")
    terminal_progress.finish()


def test_fixed_monte_carlo_progress_persists_only_final_completion() -> None:
    status = _RecordingStatus()
    progress = monte_carlo_run_progress(source="uncertainty_test", status=status)

    progress_complete(progress=progress, completed=5, max_runs=10, mode="fixed")
    progress_complete(progress=progress, completed=10, max_runs=10, mode="fixed")

    assert status.persistent == [False, True]
    assert (
        monte_carlo_completion_is_persistent(
            completed=9,
            max_runs=10,
            mode="fixed",
        )
        is False
    )
    assert (
        monte_carlo_completion_is_persistent(
            completed=10,
            max_runs=10,
            mode="fixed",
        )
        is True
    )


def test_composite_phase_index_details_and_payload(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    manifest = output_root / "logs" / "scope_manifest.json"
    entry = CompositePhaseIndexEntry(
        phase=PHASE_B2_ACC,
        function="deterministic_acc",
        status="complete",
        reuse_status="reused",
        output_root=output_root,
        summary_lines=("Source: demo",),
        info_messages=("info message",),
        warning_messages=("warning message",),
        inventory_lines=("logs: summary log.",),
    )
    empty_entry = CompositePhaseIndexEntry(
        phase=PHASE_A_LCA,
        function="external_lca",
        status="complete",
        reuse_status="computed",
        output_root=None,
    )

    assert entry.to_payload()["output_root"] == str(output_root)
    assert empty_entry.to_payload()["output_root"] is None
    ready_detail = phase_ready_detail(scope_name="aCC")
    reused_detail = phase_reused_detail(scope_name="aCC")
    assert ready_detail
    assert reused_detail
    assert "aCC" in ready_detail
    assert "aCC" in reused_detail
    assert str(output_root) in phase_ready_detail(
        scope_name="aCC",
        output_root=output_root,
    )
    assert str(output_root) in phase_reused_detail(
        scope_name="aCC",
        output_root=output_root,
    )

    index_path = write_phase_index(
        metadata_path=manifest,
        entries=[entry, empty_entry],
    )
    assert index_path == phase_index_path_for_metadata(metadata_path=manifest)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0] == {
        "phase": PHASE_B2_ACC,
        "function": "deterministic_acc",
        "status": "complete",
        "reuse_status": "reused",
        "output_root": str(output_root),
        "summary_lines": ["Source: demo"],
        "info_messages": ["info message"],
        "warning_messages": ["warning message"],
        "inventory_lines": ["logs: summary log."],
    }
    assert read_phase_index(metadata_path=manifest) == (entry, empty_entry)
    assert public_phase_reuse_status(run_status="reused_exact") == "reused"
    assert public_phase_index_reuse_status("reused") == "reused"


def test_runtime_reporting_path_and_inventory_contracts(tmp_path: Path) -> None:
    monte_carlo_path = tmp_path / "demo" / "monte_carlo" / "mc_1" / "logs" / "scope.json"
    deterministic_path = tmp_path / "demo" / "deterministic" / "logs" / "scope.json"
    logs_path = tmp_path / "demo" / "logs" / "scope.json"
    path = tmp_path / "demo" / "nested" / "file.csv"

    assert public_output_root_from_path(monte_carlo_path).name == "mc_1"
    assert public_output_root_from_path(deterministic_path).name == "deterministic"
    assert public_output_root_from_path(logs_path).name == "demo"

    set_default_repo_root(tmp_path)
    try:
        assert public_output_root_from_path(path).name == "demo"
    finally:
        clear_default_repo_root()

    lines = inventory_lines(
        [
            inventory_item(folder="results", content="row identity."),
            inventory_item(folder="results", content="row identity"),
            inventory_item(folder=" ", content="ignored"),
            inventory_item(folder="logs", content=" "),
        ]
    )
    assert len(lines) == 2
    assert lines[1].startswith("results:")
    assert "row identity" in lines[1]
    assert format_year_ranges([])
    year_range_text = format_year_ranges([2030, 2020, 2021, 2023])
    assert "2020" in year_range_text
    assert "2030" in year_range_text
    assert phase_label_for_owner("uncertainty_acc") == PHASE_B2_ACC
    assert phase_label_for_owner(None) is None
    assert plural_label(2, "file") == "files"
    assert as_sequence("method") == ("method",)
    assert as_sequence(["method"]) == ("method",)
    assert format_values(["a", ""]) == "a"
    assert format_values([]) == "none"
    assert format_report_value({"a": ["b", "c"]})
    assert format_summary_value(key="years", value=2030) == "2030"
    assert format_summary_value(key="study_period", value=[2015, 2016, 2030])
    assert format_ssp_value("2") == "SSP2"
    assert format_ssp_value("SSP2") == "SSP2"
    nested = render_summary(
        document(
            "demo",
            sections=(
                section(
                    "parent",
                    lines=("line",),
                    children=(section("child", lines=("child line",)),),
                ),
            ),
        )
    )
    assert "parent" in nested and "child" in nested


def test_phase_and_status_progress_contracts(capsys) -> None:
    phase = PhasePrinter(source="uncertainty_asr")
    phase.clear_transient()
    phase.announce("Phase custom", detail=" ")
    phase.status("   ")
    phase.show("runtime status")
    phase.log_message("   ")
    phase.complete("   ")
    phase.complete("aCC scope reused")
    phase.status("[custom_owner] running")
    phase.status("[uncertainty_acc] running", owner=None)
    phase.log_message("[uncertainty_acc] detail", persistent=False)
    phase.complete("[uncertainty_acc] aCC scope ready")
    phase.finish()
    assert capsys.readouterr().out

    null_phase = NullPhasePrinter()
    null_phase.announce("Phase A")
    null_phase.status("status")
    null_phase.show("show")
    null_phase.log_message("message")
    null_phase.complete("done")
    null_phase.finish()

    status = _RecordingStatus()
    progress = StatusProgressPrinter(
        source="long_source_name_for_status_progress",
        action="processing",
        total=0,
        status=status,
    )
    progress.begin_year(2020)
    progress.complete_year(2020)
    progress.skip_year()
    progress.show("visible")
    assert status.messages == ["visible"]

    counted = StatusProgressPrinter(
        source="io_lca",
        action="processing",
        total=2,
        status=status,
    )
    counted.begin_year("batch_a")
    counted.complete_year("batch_a")
    counted.skip_year()
    counted.log_message("persistent")
    counted.log_message("transient", persistent=False)
    counted.clear_transient()
    counted.finish()
    assert any("batch_a" in message for message in status.messages)
    assert status.persistent[-2:] == [True, False]
    assert status.cleared

    transient = YearProgressPrinter(
        source="status",
        action="message",
        total=0,
        show_timing=False,
        use_ipy_display=False,
    )
    transient.show("visible transient")
    transient.finish()
    run_progress = monte_carlo_run_progress(source="uncertainty_acc")
    run_progress.clear_transient()
    run_progress.finish()
    assert (
        visible_status_for_run_work(
            progress=run_progress,
            fallback=status,
            progress_enabled=False,
        )
        is status
    )
