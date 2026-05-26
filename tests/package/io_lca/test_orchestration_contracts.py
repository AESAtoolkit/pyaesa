import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.io_lca.contracts import fu_mapping as fu_mapping_mod
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec, resolve_fu_spec
from pyaesa.io_lca.contracts.runtime_types import IOLCAReport
from pyaesa.io_lca.compute.upstream_stages import (
    _combo_driver_vector,
    _combo_records,
    _drop_zero_rows,
    compute_upstream_rows,
)
from pyaesa.io_lca.compute.stage_linkage import dominant_parent_link_map
from pyaesa.io_lca.compute.upstream_origin import finalize_origin_rows
from pyaesa.io_lca.data.column_definitions import (
    render_origin_columns_defs,
    render_stage_columns_defs,
    write_origin_columns_defs,
    write_stage_columns_defs,
)
from pyaesa.io_lca.data.loaders import (
    load_domain_metadata,
    load_main_payload,
    load_upstream_payload,
)
from pyaesa.io_lca.data import metadata as metadata_mod
from pyaesa.io_lca.data.metadata import save_scope_manifest
from pyaesa.io_lca.data import paths as data_paths
from pyaesa.io_lca.data.paths import (
    io_metadata_path_for_source,
    origin_results_path,
    stage_results_path,
    resolve_io_lca_paths,
)
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path
from pyaesa.io_lca.orchestration.pipeline import progress
from pyaesa.io_lca.orchestration.request import domain_checks
from pyaesa.io_lca.orchestration.request import selectors
from pyaesa.io_lca.orchestration.request import validation
from pyaesa.io_lca.orchestration.request import year_resolution
from pyaesa.io_lca.orchestration.figure_support import (
    done_and_skipped_lcia_years,
    require_main_result_columns,
    resolve_io_scope,
    validate_lcia_method_coverage,
)
from pyaesa.io_lca.orchestration.pipeline.mode_runner import (
    _ensure_no_conflicting_group_indices_project,
    _ensure_no_conflicting_flat_output_scope,
    _read_signature_group_indices,
    execute_io_lca_mode,
)
from pyaesa.io_lca.orchestration.io.method_support import (
    aggregate_main,
    aggregate_origin,
    aggregate_stage,
    pending_stage_years,
    selector_combos,
    to_origin_ratio_wide,
    validate_upstream_origin_matches_main,
)
from pyaesa.io_lca.orchestration.io.method_writes import (
    _normalize_blank_identifier_values,
    write_main_year,
    write_origin_year,
    write_stage_year,
)
from pyaesa.io_lca.orchestration.reporting.summary import build_io_lca_summary
from pyaesa.io_lca.orchestration.pipeline.run_signatures import (
    build_io_lca_figure_signature,
    build_io_lca_signature,
    table_extension_for_output,
)


def _filters() -> dict[str, list[str] | None]:
    return {"r_f": ["FR", "DE"], "r_c": None, "r_p": None, "s_p": None}


def test_validation_and_year_resolution_contracts_cover_success_and_errors(
    io_lca_dummy_repo,
) -> None:
    assert (
        validation.normalize_supported_source(
            source=io_lca_dummy_repo.source,
            caller="deterministic_io_lca",
        )
        == io_lca_dummy_repo.source
    )

    with pytest.raises(ValueError):
        validation.normalize_supported_source(source=cast(Any, None), caller="deterministic_io_lca")
    with pytest.raises(ValueError):
        validation.normalize_supported_source(source="   ", caller="deterministic_io_lca")

    with pytest.raises(ValueError):
        validation.normalize_supported_source(
            source="oecd_v2025",
            caller="deterministic_io_lca",
        )

    assert validation.normalize_lcia_method_list(lcia_method="pb_lcia") == ["pb_lcia"]
    with pytest.raises(ValueError):
        validation.normalize_lcia_method_list(lcia_method=[])
    assert validation.normalize_group_indices_modes(False) == [False]
    assert validation.normalize_aggregation(
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
    ) == (False, False, None)
    assert validation.normalize_aggregation(
        agg_reg=True,
        agg_sec=False,
        agg_version="  v1  ",
    ) == (True, False, "v1")
    assert validation.normalize_io_output_format("csv") == "csv"
    assert validation.normalize_figure_output_format("png") == "png"

    with pytest.raises(ValueError):
        validation.normalize_aggregation(agg_reg=True, agg_sec=False, agg_version=None)
    with pytest.raises(ValueError):
        validation.normalize_aggregation(agg_reg=False, agg_sec=False, agg_version="v1")
    with pytest.raises(ValueError):
        validation.normalize_group_indices_modes(cast(Any, "both"))
    with pytest.raises(ValueError):
        validation.normalize_figure_output_format("gif")
    with pytest.raises(ValueError):
        validation.validate_upstream_stages(0)
    with pytest.raises(ValueError):
        validation.validate_upstream_stages(cast(Any, "bad"))
    assert validation.validate_upstream_stages(2) == 2
    with pytest.raises(ValueError):
        validation.validate_dpi(-1)
    with pytest.raises(ValueError):
        validation.validate_dpi(cast(Any, "bad"))
    assert validation.validate_dpi(300) == 300

    resolved_all = year_resolution.resolve_years_strict(
        years=None,
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    )
    assert resolved_all == [2019, 2020]
    assert year_resolution.resolve_subset_years(
        years=None,
        universe=[2020, 2019],
        label="years",
    ) == [2019, 2020]
    assert year_resolution.resolve_subset_years(
        years=[2020],
        universe=[2020, 2019],
        label="years",
    ) == [2020]
    assert year_resolution.resolve_subset_years(
        years=range(2019, 2021),
        universe=[2020, 2019],
        label="years",
    ) == [2019, 2020]
    assert (
        year_resolution.resolve_subset_years(
            years=None,
            universe=[],
            label="years",
        )
        == []
    )
    assert year_resolution.resolve_years_strict(
        years=[2019],
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        upstream_analysis=True,
    ) == [2019]
    with pytest.raises(ValueError):
        year_resolution.resolve_years_strict(
            years=[2021],
            source=io_lca_dummy_repo.source,
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )
    with pytest.raises(ValueError):
        year_resolution.resolve_years_strict(
            years=[2500],
            source=io_lca_dummy_repo.source,
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )

    with pytest.raises(ValueError):
        year_resolution.resolve_years_strict(
            years=[1800],
            source=io_lca_dummy_repo.source,
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )
    with pytest.raises(ValueError):
        year_resolution.resolve_subset_years(years=[2021], universe=[2019, 2020], label="years")


def test_resolve_years_strict_allows_selected_io_lca_years_with_non_consecutive_metadata(
    project_repo: Path,
) -> None:
    del project_repo
    metadata_path = _get_metadata_path("exiobase_3102_ixi", matrix_version=None)
    metadata_path.write_text(
        json.dumps(
            {
                "source": "exiobase_3102_ixi",
                "labels": {},
                "years": {
                    "2019": {},
                    "2021": {},
                },
            }
        ),
        encoding="utf-8",
    )

    assert year_resolution.resolve_years_strict(
        years=[2021],
        source="exiobase_3102_ixi",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    ) == [2021]
    assert year_resolution.resolve_years_strict(
        years=None,
        source="exiobase_3102_ixi",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    ) == [2019, 2021]
    with pytest.raises(ValueError):
        year_resolution.resolve_years_strict(
            years=[2020],
            source="exiobase_3102_ixi",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )


def test_resolve_years_strict_rejects_empty_metadata(
    project_repo: Path,
) -> None:
    del project_repo
    empty_metadata_path = _get_metadata_path("exiobase_3102_ixi", matrix_version="empty")
    empty_metadata_path.write_text(
        json.dumps(
            {
                "source": "exiobase_3102_ixi",
                "labels": {},
                "years": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        year_resolution.resolve_years_strict(
            years=None,
            source="exiobase_3102_ixi",
            agg_version="empty",
            agg_reg=False,
            agg_sec=False,
        )


def test_domain_checks_cover_aggregated_branch_and_fu_constraints(tmp_path: Path) -> None:
    metadata_path = tmp_path / "aggregated" / "metadata.json"

    domain_checks.require_aggregated_branch(
        source="exiobase_396_ixi",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata_path=metadata_path,
        methods=["pb_lcia"],
        years=[2019],
    )

    with pytest.raises(ValueError):
        domain_checks.require_aggregated_branch(
            source="exiobase_396_ixi",
            agg_version="v1",
            agg_reg=True,
            agg_sec=False,
            metadata_path=metadata_path,
            methods=["pb_lcia"],
            years=[2019],
        )
    with pytest.raises(ValueError):
        domain_checks.require_aggregated_branch(
            source="exiobase_396_ixi",
            agg_version="v1",
            agg_reg=True,
            agg_sec=False,
            metadata_path=metadata_path,
            methods=["pb_lcia"],
            years=2019,
        )
    with pytest.raises(ValueError):
        domain_checks.require_aggregated_branch(
            source="exiobase_396_ixi",
            agg_version="v1",
            agg_reg=True,
            agg_sec=False,
            metadata_path=metadata_path,
            methods=["pb_lcia"],
            years=None,
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("{}", encoding="utf-8")
    domain_checks.require_aggregated_branch(
        source="exiobase_396_ixi",
        agg_version="v1",
        agg_reg=True,
        agg_sec=False,
        metadata_path=metadata_path,
        methods=["pb_lcia"],
        years=[2019],
    )

    spec_l2_pba = resolve_fu_spec(fu_code="L2.a.c")
    domain_checks.validate_upstream_supported(spec=spec_l2_pba, upstream_analysis=False)
    with pytest.raises(ValueError):
        domain_checks.validate_upstream_supported(spec=spec_l2_pba, upstream_analysis=True)

    spec_l2_td = resolve_fu_spec(fu_code="L2.a.b")
    domain_checks.validate_group_indices_supported(spec=spec_l2_td, group_indices=False)
    with pytest.raises(ValueError):
        domain_checks.validate_group_indices_supported(spec=spec_l2_td, group_indices=True)

    domain_checks.validate_group_indices_requires_multi_selection(
        group_indices=False,
        has_multi_indices=False,
    )
    with pytest.raises(ValueError):
        domain_checks.validate_group_indices_requires_multi_selection(
            group_indices=True,
            has_multi_indices=False,
        )
    assert spec_l2_td.family == "td"
    assert spec_l2_td.upstream_driver == "x_to_rc"
    assert spec_l2_pba.family == "pba"
    assert spec_l2_pba.upstream_supported is False
    assert fu_mapping_mod._family_from_kind(lcia_kind="CBA_TD") == "td"  # noqa: SLF001
    assert fu_mapping_mod._family_from_kind(lcia_kind="PBA") == "pba"  # noqa: SLF001
    assert (
        fu_mapping_mod._lcia_matrix_key_for_fu(  # noqa: SLF001
            fu_code="L1.a",
            lcia_kind="CBA_FD",
        )
        == resolve_fu_spec(fu_code="L1.a").lcia_matrix_key
    )
    assert (
        fu_mapping_mod._lcia_matrix_key_for_fu(  # noqa: SLF001
            fu_code="L2.a.a",
            lcia_kind="CBA_FD",
        )
        == resolve_fu_spec(fu_code="L2.a.a").lcia_matrix_key
    )
    with pytest.raises(ValueError):
        resolve_fu_spec(fu_code="L2.z.a")


def test_progress_contracts_cover_formatting_and_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert progress.source_prefix(source=" exiobase_396_ixi ") == "[exiobase_396_ixi]"
    assert progress.format_year_ranges_with_count([2020, 2019, 2019]) == "2019-2020 (2 year(s))"
    assert progress.format_method_labels([" pb_lcia ", "", "aa"]) == "aa, pb_lcia"
    assert progress.format_method_labels([" ", ""]) == "none"
    assert progress.format_indices_label(_filters()) == "r_f=FR+DE"
    assert progress.format_indices_label({"r_p": None, "s_p": None, "r_c": None, "r_f": None}) == (
        "all"
    )
    assert (
        progress.format_indices_label(
            {"r_p": ["R1", "R2"], "s_p": ["S1"], "r_c": None, "r_f": ["FR"]}
        )
        == "r_p=R1+R2, s_p=S1, r_f=FR"
    )

    progress.io_lca_banner(
        source="exiobase_396_ixi",
        years=[2019],
        methods=[],
        fu_code="L1.a",
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
        upstream_analysis=False,
        upstream_stages=1,
    )
    progress.io_lca_banner(
        source="exiobase_396_ixi",
        years=[2020, 2019, 2019],
        methods=["pb_lcia", "aa"],
        fu_code="L1.a",
        filters=_filters(),
        upstream_analysis=True,
        upstream_stages=3,
    )
    progress.io_lca_mode_banner(
        source="exiobase_396_ixi",
        fu_code="L1.a",
        filters=_filters(),
        mode_tag=None,
    )
    progress.io_lca_mode_banner(
        source="exiobase_396_ixi",
        fu_code="L1.a",
        filters=_filters(),
        mode_tag="both",
    )
    output = capsys.readouterr().out
    assert output


def test_mode_runner_flat_output_scope_contracts_cover_malformed_scopes(
    io_lca_dummy_repo,
) -> None:
    filters = _filters()
    with pytest.raises(ValueError):
        _ensure_no_conflicting_flat_output_scope(  # noqa: SLF001
            log_payload={
                "arguments": {
                    "fu_code": "L1.a",
                    "selectors": {"r_f": ["US"]},
                }
            },
            fu_code="L1.a",
            filters=filters,
        )


def test_runtime_types_path_logging_origin_and_selector_contracts(
    io_lca_dummy_repo,
    tmp_path: Path,
) -> None:
    report_with_lines = IOLCAReport(
        source="exiobase_396_ixi",
        fu_code="L1.a",
        years=[2019],
        lcia_methods=["pb_lcia"],
        main_result_paths=[],
        origin_paths=[],
        stage_paths=[],
        skipped_method_years={},
        metadata_path=tmp_path / "meta_with_lines.json",
        summary_lines=["Line 1", "Line 2"],
    )
    assert str(report_with_lines) == "Line 1\nLine 2"
    assert repr(report_with_lines) == str(report_with_lines)

    empty_origin = finalize_origin_rows(
        origin_rows=pd.DataFrame(),
        selector_axes=("r_f",),
    )
    assert list(empty_origin.columns) == [
        "year",
        "impact",
        "origin_r_p",
        "origin_s_p",
        "impact_unit",
        "r_f",
        "lca_value",
    ]

    aggregated_paths = data_paths.resolve_io_lca_paths(
        project_name="io_lca_path_helpers",
        agg_reg=True,
        agg_sec=False,
        agg_version=" Demo / Version ",
    )
    assert aggregated_paths.source_version_token == "Demo_Version"
    assert aggregated_paths.lca_root == aggregated_paths.project_base / "A_lca" / "io_lca"
    assert (
        data_paths.source_scope_root_for_source(
            paths=aggregated_paths,
            source="exiobase 396 ixi",
        ).name
        == "exiobase_396_ixi__Demo_Version"
    )
    assert data_paths.deterministic_scope_metadata_paths(paths=aggregated_paths) == []

    assert (
        selectors.has_multi_selected_indices({"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": None})
        is False
    )
    resolved_filters, tag = selectors.resolve_selectors(
        spec=resolve_fu_spec(fu_code="L1.a"),
        r_f=" FR ",
        r_c=None,
        r_p=None,
        s_p=None,
    )
    assert resolved_filters["r_f"] == ["FR"]
    assert tag == "r_f-FR"

    metadata, _metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    labels = metadata["labels"]
    assert isinstance(labels, dict)
    sectors_used = labels["sectors_used"]
    assert isinstance(sectors_used, list)
    selectors.validate_selector_labels(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        filters={"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": None},
    )
    selectors.validate_selector_labels(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        filters={"r_f": ["FR"], "r_c": None, "r_p": None, "s_p": [str(sectors_used[0])]},
    )


def test_run_signatures_and_scope_contracts_cover_matching_and_reuse_failures(
    tmp_path: Path,
) -> None:
    filters = _filters()
    signature = build_io_lca_signature(
        project_name="proj",
        source="exiobase_396_ixi",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        years=[2019, 2020],
        methods=["pb_lcia"],
        fu_code="L1.a",
        filters=filters,
        upstream_analysis=False,
        upstream_stages=1,
        group_indices=False,
        output_format="csv",
    )
    figure_signature = build_io_lca_figure_signature(
        project_name="proj",
        source="exiobase_396_ixi",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        years=[2019],
        methods=["pb_lcia"],
        fu_code="L1.a",
        filters=filters,
        group_indices=False,
        dpi=300,
        output_format="png",
        io_output_format="csv",
    )
    assert signature["selectors"]["r_f"] == ["FR", "DE"]
    assert signature["upstream_analysis"] is False
    assert figure_signature["dpi"] == 300
    assert table_extension_for_output("csv") == "csv"
    assert table_extension_for_output("pickle") == "pickle"
    assert table_extension_for_output("parquet") == "parquet"

    scope_path = tmp_path / "scope.csv"
    scope_path.write_text("ok", encoding="utf-8")
    payload = {
        "arguments": signature,
        "complete": True,
        "paths_written": [str(scope_path)],
        "status": {},
    }
    io_output_format, scope = resolve_io_scope(
        io_log_payload=payload,
        project_name="proj",
        source="exiobase_396_ixi",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        years=[2019, 2020],
        lcia_methods=["pb_lcia"],
        fu_code="L1.a",
        filters=filters,
        group_indices=False,
    )
    assert io_output_format == "csv"
    assert scope is payload

    incomplete_payload = {
        "arguments": signature,
        "complete": False,
        "paths_written": [str(scope_path)],
        "status": {},
    }
    with pytest.raises(ValueError):
        resolve_io_scope(
            io_log_payload=incomplete_payload,
            project_name="proj",
            source="exiobase_396_ixi",
            agg_reg=False,
            agg_sec=False,
            agg_version=None,
            years=[2019, 2020],
            lcia_methods=["pb_lcia"],
            fu_code="L1.a",
            filters=filters,
            group_indices=False,
        )

    with pytest.raises(ValueError):
        resolve_io_scope(
            io_log_payload=metadata_mod.load_scope_manifest(
                path=tmp_path / "missing_io_manifest.json",
                function_name="deterministic_io_lca",
            ),
            project_name="proj",
            source="exiobase_396_ixi",
            agg_reg=False,
            agg_sec=False,
            agg_version=None,
            years=[2019, 2020],
            lcia_methods=["pb_lcia"],
            fu_code="L1.a",
            filters=filters,
            group_indices=False,
        )

    done, skipped = done_and_skipped_lcia_years(
        scope={
            "status": {
                "main": {
                    "pb_lcia": {
                        "years_done": [2019, "2020"],
                        "years_skipped": {"2021": "skip"},
                    }
                }
            }
        },
        lcia_method="pb_lcia",
    )
    assert done == {2019, 2020}
    assert skipped == {2021}
    assert done_and_skipped_lcia_years(
        scope={"status": {"main": {"pb_lcia": {"years_done": [], "years_skipped": {}}}}},
        lcia_method="pb_lcia",
    ) == (set(), set())

    validate_lcia_method_coverage(
        io_scope={
            "status": {
                "main": {
                    "pb_lcia": {
                        "years_done": [2019],
                        "years_skipped": {"2020": "skip"},
                    }
                }
            }
        },
        lcia_method="pb_lcia",
        years=[2019, 2020],
    )
    with pytest.raises(ValueError):
        validate_lcia_method_coverage(
            io_scope={
                "status": {
                    "main": {
                        "pb_lcia": {
                            "years_done": [2019],
                            "years_skipped": {"2020": "skip"},
                        }
                    }
                }
            },
            lcia_method="pb_lcia",
            years=[2019, 2021],
        )

    frame = pd.DataFrame(
        {
            "year": [2019],
            "impact": ["AAL"],
            "lca_value": [1.0],
            "impact_unit": ["kg"],
            "lcia_method": ["pb_lcia"],
            "r_f": ["FR"],
        }
    )
    require_main_result_columns(frame=frame, lcia_method="pb_lcia", selector_axes=("r_f",))
    with pytest.raises(ValueError):
        require_main_result_columns(
            frame=frame.drop(columns=["impact_unit"]),
            lcia_method="pb_lcia",
            selector_axes=("r_f",),
        )


def test_io_lca_method_support_contracts_and_pending_stage_years(
    io_lca_dummy_repo,
) -> None:
    spec = resolve_fu_spec(fu_code="L1.a")
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    payload, _ = load_main_payload(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata=metadata,
        metadata_path=metadata_path,
        year=2019,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=spec,
    )
    assert payload is not None

    main = pd.DataFrame(
        {
            "year": [2019, 2019],
            "lcia_method": ["pb_lcia", "pb_lcia"],
            "impact": ["AAL", "AAL"],
            "impact_unit": ["kg", "kg"],
            "lca_value": [1.0, 2.0],
        }
    )
    main_aggregate = aggregate_main(main)
    assert main_aggregate.loc[0, "lca_value"] == pytest.approx(3.0)
    assert main_aggregate.columns.tolist() == [
        "lcia_method",
        "year",
        "impact",
        "lca_value",
        "impact_unit",
    ]
    origin_aggregate = aggregate_origin(
        pd.DataFrame(
            {
                "year": [2019, 2019],
                "lcia_method": ["pb_lcia", "pb_lcia"],
                "impact": ["AAL", "AAL"],
                "origin_r_p": ["FR", "FR"],
                "origin_s_p": ["A", "A"],
                "impact_unit": ["kg", "kg"],
                "lca_value": [1.0, 2.0],
            }
        )
    )
    assert origin_aggregate.loc[0, "lca_value"] == pytest.approx(3.0)
    assert origin_aggregate.columns.tolist() == [
        "year",
        "impact",
        "origin_r_p",
        "origin_s_p",
        "impact_unit",
        "lca_value",
    ]
    stage_aggregate = aggregate_stage(
        pd.DataFrame(
            {
                "lcia_method": ["pb_lcia", "pb_lcia"],
                "stage": ["s1", "s1"],
                "stage_r_p": ["FR", "FR"],
                "stage_s_p": ["A", "A"],
                "linked_from_stage": ["root", "root"],
                "linked_from_r_p": ["FR", "FR"],
                "linked_from_s_p": ["A", "A"],
                "impact": ["AAL", "AAL"],
                "impact_unit": ["kg", "kg"],
                "direct_at_stage": [1.0, 2.0],
                "embedded_from_deeper_stages": [3.0, 4.0],
                "stage_total": [4.0, 6.0],
            }
        )
    )
    assert stage_aggregate.loc[0, "stage_total"] == pytest.approx(10.0)
    assert stage_aggregate.columns.tolist() == [
        "stage",
        "stage_r_p",
        "stage_s_p",
        "linked_from_stage",
        "linked_from_r_p",
        "linked_from_s_p",
        "impact",
        "impact_unit",
        "direct_at_stage",
        "embedded_from_deeper_stages",
        "stage_total",
    ]

    validate_upstream_origin_matches_main(
        main_frame=pd.DataFrame(),
        origin_frame=pd.DataFrame(),
        selector_axes=tuple(),
        lcia_method="pb_lcia",
    )
    validate_upstream_origin_matches_main(
        main_frame=main,
        origin_frame=pd.DataFrame(),
        selector_axes=tuple(),
        lcia_method="pb_lcia",
    )
    with pytest.raises(ValueError):
        validate_upstream_origin_matches_main(
            main_frame=main.drop(columns=["impact_unit"]),
            origin_frame=pd.DataFrame(
                {
                    "year": [2019],
                    "lcia_method": ["pb_lcia"],
                    "impact": ["AAL"],
                    "impact_unit": ["kg"],
                    "lca_value": [1.0],
                }
            ),
            selector_axes=tuple(),
            lcia_method="pb_lcia",
        )
    with pytest.raises(ValueError):
        validate_upstream_origin_matches_main(
            main_frame=main,
            origin_frame=pd.DataFrame(
                {
                    "year": [2019],
                    "lcia_method": ["pb_lcia"],
                    "impact": ["AAL"],
                    "impact_unit": ["kg"],
                    "lca_value": [9.0],
                }
            ),
            selector_axes=tuple(),
            lcia_method="pb_lcia",
        )
    with pytest.raises(ValueError):
        to_origin_ratio_wide(frame=pd.DataFrame({"impact": ["AAL"]}), selector_axes=tuple())
    assert to_origin_ratio_wide(
        frame=pd.DataFrame(
            {
                "impact": ["AAL"],
                "origin_r_p": ["FR"],
                "origin_s_p": ["A"],
                "impact_unit": ["kg"],
            }
        ),
        selector_axes=tuple(),
    ).equals(
        pd.DataFrame(
            {
                "impact": ["AAL"],
                "origin_r_p": ["FR"],
                "origin_s_p": ["A"],
                "impact_unit": ["kg"],
            }
        )
    )

    ratio = to_origin_ratio_wide(
        frame=pd.DataFrame(
            {
                "impact": ["AAL"],
                "origin_r_p": ["FR"],
                "origin_s_p": ["A"],
                "impact_unit": ["kg"],
                "2019": [0.0],
            }
        ),
        selector_axes=tuple(),
    )
    assert ratio.loc[0, "2019"] == pytest.approx(0.0)

    selector_rows = selector_combos(
        payload=payload,
        spec=IOLCAFUSpec(
            fu_code="manual",
            level="L1",
            family="fd",
            lcia_matrix_key="e_cba_fd_reg",
            selector_axes=tuple(),
            upstream_driver="y_fd",
            upstream_supported=True,
            fy_relevant=True,
        ),
        lcia_method=io_lca_dummy_repo.lcia_method,
        filters=_filters(),
    )
    assert selector_rows.columns.tolist() == []
    assert selector_rows.to_dict(orient="records") == []

    non_empty_selector_rows = selector_combos(
        payload=payload,
        spec=spec,
        lcia_method=io_lca_dummy_repo.lcia_method,
        filters={"r_f": ["ZZ"], "r_c": None, "r_p": None, "s_p": None},
    )
    assert non_empty_selector_rows.columns.tolist() == ["r_f"]
    assert non_empty_selector_rows.empty
    filled_selector_rows = selector_combos(
        payload=payload,
        spec=spec,
        lcia_method=io_lca_dummy_repo.lcia_method,
        filters=_filters(),
    )
    assert filled_selector_rows.columns.tolist() == ["r_f"]
    assert sorted(filled_selector_rows["r_f"].astype(str).tolist()) == ["DE", "FR"]

    paths = resolve_io_lca_paths(
        project_name="pending_stage_years",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
    )
    stage_path = stage_results_path(
        paths=paths,
        source=io_lca_dummy_repo.source,
        lcia_method=io_lca_dummy_repo.lcia_method,
        year=2019,
        extension="csv",
    )
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("stage", encoding="utf-8")
    second_stage_path = stage_results_path(
        paths=paths,
        source=io_lca_dummy_repo.source,
        lcia_method=io_lca_dummy_repo.lcia_method,
        year=2020,
        extension="csv",
    )
    second_stage_path.write_text("stage", encoding="utf-8")
    pending = pending_stage_years(
        years=[2019, 2020, 2021],
        existing_years=[2019],
        paths=paths,
        source=io_lca_dummy_repo.source,
        lcia_method=io_lca_dummy_repo.lcia_method,
        extension="csv",
    )
    assert pending == [2021]
    stage_2021_path = stage_results_path(
        paths=paths,
        source=io_lca_dummy_repo.source,
        lcia_method=io_lca_dummy_repo.lcia_method,
        year=2021,
        extension="csv",
    )
    stage_2021_path.write_text("stage", encoding="utf-8")
    assert (
        pending_stage_years(
            years=[2021],
            existing_years=[],
            paths=paths,
            source=io_lca_dummy_repo.source,
            lcia_method=io_lca_dummy_repo.lcia_method,
            extension="csv",
        )
        == []
    )


def test_io_lca_column_definition_contracts_cover_year_duplicates_and_writes(
    tmp_path: Path,
) -> None:
    origin_text = render_origin_columns_defs(
        columns=[
            "2019",
            "2019",
            "2019.5",
            "origin_r_p",
            "origin_s_p",
            "impact_unit",
            "",
            "custom_selector",
        ]
    )
    origin_words = " ".join(origin_text.split())
    assert (
        origin_words.count(
            "2019 - Year column. In absolute tables: absolute contribution value. "
            "In ratio tables: contribution share in [0, 1]."
        )
        == 1
    )
    assert (
        origin_text.count("Selector column inherited from the FU contract and current run filters.")
        == 3
    )
    origin_path = write_origin_columns_defs(
        path=tmp_path / "defs" / "origin_columns.txt",
        columns=["impact", "2019", "2019"],
    )
    assert origin_path.read_text(encoding="utf-8") == render_origin_columns_defs(
        columns=["impact", "2019", "2019"]
    )

    stage_text = render_stage_columns_defs(
        columns=[
            "year",
            "year",
            "stage",
            "stage_r_p",
            "stage_s_p",
            "linked_from_stage",
            "linked_from_r_p",
            "linked_from_s_p",
            "impact",
            "direct_at_stage",
            "embedded_from_deeper_stages",
            "impact_unit",
            "custom_stage",
        ]
    )
    stage_words = " ".join(stage_text.split())
    assert stage_text.count("year\n- Studied year.\n") == 1
    assert (
        stage_words.count(
            "stage - Supply-chain stage label (`n`, `n-1`, ...). "
            "The `direct_final_demand_FY` row is the separate F_Y component."
        )
        == 1
    )
    assert (
        "custom_stage\n- Selector column inherited from the FU contract and current run filters.\n"
        in stage_text
    )
    stage_path = write_stage_columns_defs(
        path=tmp_path / "defs" / "stage_columns.txt",
        columns=["year", "stage", "stage"],
    )
    assert stage_path.read_text(encoding="utf-8") == render_stage_columns_defs(
        columns=["year", "stage", "stage"]
    )
    assert render_stage_columns_defs(columns=[]) == "\n"
    assert all(len(line) <= 100 for line in origin_text.splitlines())
    assert all(len(line) <= 100 for line in stage_text.splitlines())


def test_upstream_stage_contracts_cover_combo_selection_and_stage_variants(
    io_lca_dummy_repo,
) -> None:
    assert _drop_zero_rows(
        frame=pd.DataFrame(columns=["lca_value"]),
        value_columns=["lca_value"],
    ).empty

    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    pba_spec = resolve_fu_spec(fu_code="L1.b")
    pba_main, _ = load_main_payload(
        source=io_lca_dummy_repo.source,
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
        metadata=metadata,
        metadata_path=metadata_path,
        year=2019,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=pba_spec,
    )
    assert pba_main is not None
    pba_upstream = load_upstream_payload(
        source=io_lca_dummy_repo.source,
        saved_dir=pba_main.saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=pba_spec,
    )

    empty_pba_combos = pd.DataFrame(columns=list(pba_spec.selector_axes))
    assert _combo_records(
        combos=empty_pba_combos,
        selector_axes=pba_spec.selector_axes,
    ) == [{"r_p": None}]
    assert _combo_driver_vector(
        driver=pba_upstream.driver_matrix,
        spec=pba_spec,
        combo={"r_p": float("nan")},
    ).tolist() == [10.0, 20.0]
    assert _combo_driver_vector(
        driver=pba_upstream.driver_matrix,
        spec=pba_spec,
        combo={"r_p": "ZZ"},
    ).tolist() == [0.0, 0.0]

    pba_stage_rows, pba_origin_rows = compute_upstream_rows(
        year=2019,
        spec=pba_spec,
        combos=empty_pba_combos,
        payload=pba_upstream,
        upstream_stages=1,
        unit_by_impact=pba_main.unit_by_impact,
        emit_stage_rows=False,
    )
    assert pba_stage_rows.empty
    assert list(pba_stage_rows.columns) == [
        "r_p",
        "stage",
        "stage_r_p",
        "stage_s_p",
        "linked_from_stage",
        "linked_from_r_p",
        "linked_from_s_p",
        "impact",
        "impact_unit",
        "direct_at_stage",
        "embedded_from_deeper_stages",
        "stage_total",
    ]
    assert not pba_origin_rows.empty
    assert set(pba_origin_rows["origin_r_p"]) == {"FR", "DE", "F_Y"}
    assert set(pba_origin_rows["origin_s_p"]) == {"A", "B", "F_Y"}

    l2_fd_spec = IOLCAFUSpec(
        fu_code="manual_l2_fd",
        level="L2",
        family="fd",
        lcia_matrix_key="e_cba_fd_rp_sp_rf",
        selector_axes=("r_p", "s_p", "r_f"),
        upstream_driver="y_fd",
        upstream_supported=True,
        fy_relevant=False,
    )
    l2_fd_upstream = load_upstream_payload(
        source=io_lca_dummy_repo.source,
        saved_dir=pba_main.saved_dir,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_spec=l2_fd_spec,
    )
    assert l2_fd_upstream.fy_matrix is None

    l2_stage_rows, l2_origin_rows = compute_upstream_rows(
        year=2019,
        spec=l2_fd_spec,
        combos=pd.DataFrame(
            [
                {"r_p": "FR", "s_p": "A", "r_f": "FR"},
                {"r_p": "DE", "s_p": "B", "r_f": "DE"},
            ]
        ),
        payload=l2_fd_upstream,
        upstream_stages=1,
        unit_by_impact=pba_main.unit_by_impact,
        emit_stage_rows=True,
    )
    assert not l2_stage_rows.empty
    assert set(l2_stage_rows["r_f"]) == {"FR", "DE"}
    assert set(l2_stage_rows["stage"]) == {"n"}
    assert set(l2_origin_rows["r_f"]) == {"FR", "DE"}
    assert "F_Y" not in set(l2_origin_rows["origin_r_p"])


def test_stage_linkage_contracts_cover_scalar_tokens_empty_chunks_and_zero_links() -> None:
    linked = dominant_parent_link_map(
        a_matrix=pd.DataFrame(
            [[0.0, 2.0], [0.0, 0.0]],
            index=["child_scalar", ("FR", "A")],
            columns=[("DE", "C"), ("US", "B")],
        ),
        q_prev=pd.Series([1.0, 2.0], index=[("DE", "C"), ("US", "B")]),
        eps=1e-12,
    )
    assert linked[("child_scalar", "")] == ("US", "B")
    assert linked[("FR", "A")] == ("", "")

    assert (
        dominant_parent_link_map(
            a_matrix=pd.DataFrame(index=[("FR", "A")]),
            q_prev=pd.Series(dtype=float),
            eps=1e-12,
        )
        == {}
    )


def test_io_lca_method_write_contracts_cover_canonicalization_and_errors(
    tmp_path: Path,
) -> None:
    paths = resolve_io_lca_paths(
        project_name="io_lca_writes",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
    )

    main_rows = pd.DataFrame()
    assert (
        write_main_year(
            year_main_rows=main_rows,
            paths=paths,
            source="exiobase_396_ixi",
            lcia_method="pb_lcia",
            extension="csv",
            output_format="csv",
            effective_selector_axes=tuple(),
            written_main=[],
        )
        is None
    )

    main_frame = pd.DataFrame(
        {
            "year": [2019],
            "lcia_method": ["pb_lcia"],
            "impact": ["AAL"],
            "impact_unit": ["kg"],
            "lca_value": [1.0],
        }
    )
    written_main: list[Path] = []
    merged_main = write_main_year(
        year_main_rows=main_frame,
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_main=written_main,
    )
    assert merged_main is not None
    assert merged_main.loc[0, "lca_value"] == pytest.approx(1.0)
    assert len(written_main) == 1

    merged_main_repeat = write_main_year(
        year_main_rows=main_frame,
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_main=written_main,
    )
    assert merged_main_repeat is not None
    assert len(written_main) == 1

    origin_rows = pd.DataFrame(
        {
            "year": [2019],
            "lcia_method": ["pb_lcia"],
            "impact": ["AAL"],
            "origin_r_p": ["F_Y"],
            "origin_s_p": [""],
            "impact_unit": ["kg"],
            "lca_value": [1.0],
        }
    )
    assert (
        write_origin_year(
            year_origin_rows=origin_rows.iloc[0:0].copy(),
            main_for_check=main_frame,
            lcia_method="pb_lcia",
            paths=paths,
            source="exiobase_396_ixi",
            extension="csv",
            output_format="csv",
            effective_selector_axes=tuple(),
            written_origin=[],
        )
        is None
    )

    written_origin: list[Path] = []
    write_origin_year(
        year_origin_rows=origin_rows,
        main_for_check=main_frame,
        lcia_method="pb_lcia",
        paths=paths,
        source="exiobase_396_ixi",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_origin=written_origin,
    )
    origin_path = origin_results_path(
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        extension="csv",
    )
    assert origin_path.exists()
    origin_frame = pd.read_csv(origin_path)
    assert origin_frame.columns.tolist() == [
        "impact",
        "origin_r_p",
        "origin_s_p",
        "impact_unit",
        "2019",
    ]
    assert origin_frame.loc[0, "origin_s_p"] == "F_Y"
    assert len(written_origin) == 2

    write_origin_year(
        year_origin_rows=origin_rows,
        main_for_check=main_frame,
        lcia_method="pb_lcia",
        paths=paths,
        source="exiobase_396_ixi",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_origin=written_origin,
    )
    assert len(written_origin) == 2

    direct_origin_rows = origin_rows.assign(origin_r_p=["FR"], origin_s_p=["A"])
    write_origin_year(
        year_origin_rows=direct_origin_rows,
        main_for_check=main_frame,
        lcia_method="pb_lcia",
        paths=paths,
        source="exiobase_396_ixi",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_origin=written_origin,
    )
    origin_frame = pd.read_csv(origin_path)
    direct_row = origin_frame.loc[origin_frame["origin_r_p"] == "FR"].reset_index(drop=True)
    assert direct_row.loc[0, "origin_s_p"] == "A"

    with pytest.raises(ValueError):
        write_origin_year(
            year_origin_rows=origin_rows.assign(lca_value=[2.0]),
            main_for_check=main_frame,
            lcia_method="pb_lcia",
            paths=paths,
            source="exiobase_396_ixi",
            extension="csv",
            output_format="csv",
            effective_selector_axes=tuple(),
            written_origin=[],
        )

    assert (
        write_stage_year(
            year=2019,
            year_stage_rows=pd.DataFrame(),
            paths=paths,
            source="exiobase_396_ixi",
            lcia_method="pb_lcia",
            extension="csv",
            output_format="csv",
            effective_selector_axes=tuple(),
            written_stage=[],
        )
        is None
    )

    written_stage: list[Path] = []
    write_stage_year(
        year=2019,
        year_stage_rows=pd.DataFrame(
            {
                "lcia_method": ["pb_lcia"],
                "stage": ["root"],
                "stage_r_p": ["FR"],
                "stage_s_p": ["A"],
                "linked_from_stage": ["root"],
                "linked_from_r_p": ["FR"],
                "linked_from_s_p": ["A"],
                "impact": ["AAL"],
                "impact_unit": ["kg"],
                "direct_at_stage": [1.0],
                "embedded_from_deeper_stages": [0.0],
                "stage_total": [1.0],
            }
        ),
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_stage=written_stage,
    )
    stage_path = stage_results_path(
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        year=2019,
        extension="csv",
    )
    assert stage_path.exists()
    assert pd.read_csv(stage_path).columns.tolist() == [
        "stage",
        "stage_r_p",
        "stage_s_p",
        "linked_from_stage",
        "linked_from_r_p",
        "linked_from_s_p",
        "impact",
        "impact_unit",
        "direct_at_stage",
        "embedded_from_deeper_stages",
        "stage_total",
    ]
    assert len(written_stage) == 1

    normalized_ids = _normalize_blank_identifier_values(  # noqa: SLF001
        frame=pd.DataFrame(
            {
                "origin_r_p": ["F_Y"],
                "origin_s_p": [""],
                "year": [2019],
            }
        ),
        id_columns=["origin_r_p", "origin_s_p", "year", "missing_column"],
    )
    assert pd.isna(normalized_ids.loc[0, "origin_s_p"])
    assert normalized_ids.loc[0, "year"] == 2019

    write_stage_year(
        year=2019,
        year_stage_rows=pd.DataFrame(
            {
                "year": [2019],
                "lcia_method": ["pb_lcia"],
                "stage": ["root"],
                "stage_r_p": ["FR"],
                "stage_s_p": ["A"],
                "linked_from_stage": ["root"],
                "linked_from_r_p": ["FR"],
                "linked_from_s_p": ["A"],
                "impact": ["AAL"],
                "impact_unit": ["kg"],
                "direct_at_stage": [1.0],
                "embedded_from_deeper_stages": [0.0],
                "stage_total": [1.0],
            }
        ),
        paths=paths,
        source="exiobase_396_ixi",
        lcia_method="pb_lcia",
        extension="csv",
        output_format="csv",
        effective_selector_axes=tuple(),
        written_stage=written_stage,
    )
    assert len(written_stage) == 1


def test_mode_runner_and_reporting_contracts_cover_refresh_and_summary_variants(
    io_lca_dummy_repo,
    tmp_path: Path,
) -> None:
    assert _read_signature_group_indices({"group_indices": True}) is True  # noqa: SLF001

    paths = resolve_io_lca_paths(
        project_name="io_lca_mode_runner_helpers",
        agg_reg=False,
        agg_sec=False,
        agg_version="",
    )
    current_metadata_path = io_metadata_path_for_source(
        paths=paths,
        source=io_lca_dummy_repo.source,
    )

    other_metadata_path = io_metadata_path_for_source(paths=paths, source="exiobase_396_pxp")
    other_signature = {"group_indices": True}
    save_scope_manifest(
        path=other_metadata_path,
        payload={
            "function": "deterministic_io_lca",
            "arguments": other_signature,
            "timestamp": "2026-04-11T00:00:00+00:00",
            "complete": True,
            "paths_written": [],
            "status": {"main": {}, "origin": {}, "stages": {}, "figures": {}},
        },
    )
    with pytest.raises(ValueError):
        _ensure_no_conflicting_group_indices_project(  # noqa: SLF001
            paths=paths,
            current_metadata_path=current_metadata_path,
            log_payload=metadata_mod.load_scope_manifest(
                path=tmp_path / "missing_scope_manifest.json",
                function_name="deterministic_io_lca",
            ),
            group_indices=False,
        )
    other_metadata_path.unlink()

    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        agg_version=None,
    )
    mode_result = execute_io_lca_mode(
        project_name="io_lca_mode_runner_helpers",
        source=io_lca_dummy_repo.source,
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        methods=[io_lca_dummy_repo.lcia_method],
        spec=resolve_fu_spec(fu_code="L1.a"),
        filters=_filters(),
        metadata=metadata,
        domain_metadata_path=metadata_path,
        resolved_years=[2019],
        upstream_analysis=False,
        stages=1,
        stage_outputs_enabled=False,
        group_indices=False,
        output_format="csv",
        refresh=True,
        has_multi_indices=True,
    )
    assert mode_result.main_paths
    assert mode_result.origin_paths == []
    assert mode_result.stage_paths == []
    assert mode_result.skipped_method_years == {}

    summary_lines = build_io_lca_summary(
        source=io_lca_dummy_repo.source,
        output_root=tmp_path / "multi",
        resolved_years=[2019, 2020],
        covered_main_years={2019},
        covered_origin_years={2019},
        covered_stage_years={2019},
        skipped_method_years={"pb_lcia__group_indices": {2020: "extension missing"}},
        group_indices=True,
        upstream_analysis=True,
        stage_outputs_enabled=True,
        reuse_status="computed",
        lca_results_dirs={tmp_path / "main_a", tmp_path / "main_b"},
        origin_dirs={tmp_path / "origin_a", tmp_path / "origin_b"},
        stages_dirs={tmp_path / "stage_a", tmp_path / "stage_b"},
        figure_paths=[tmp_path / "fig_a" / "plot.png"],
    )
    assert len(summary_lines) > 1

    single_figure_summary = build_io_lca_summary(
        source=io_lca_dummy_repo.source,
        output_root=tmp_path / "single",
        resolved_years=[2019],
        covered_main_years={2019},
        covered_origin_years=set(),
        covered_stage_years=set(),
        skipped_method_years={},
        group_indices=False,
        upstream_analysis=False,
        stage_outputs_enabled=False,
        reuse_status="computed",
        lca_results_dirs={tmp_path / "main_only"},
        origin_dirs=set(),
        stages_dirs=set(),
        figure_paths=[tmp_path / "fig_only" / "plot.png"],
    )
    assert len(single_figure_summary) > 1
