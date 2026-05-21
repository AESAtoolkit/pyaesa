from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.io_lca.compute import upstream_long_rows as long_rows_mod
from pyaesa.io_lca.data import column_definitions as column_defs_mod
from pyaesa.io_lca.data import metadata as metadata_mod
from pyaesa.io_lca.data import paths as paths_mod


def test_io_lca_upstream_long_row_contracts_cover_pairing_shape_and_empty_paths() -> None:
    multi_index = pd.MultiIndex.from_tuples([("FR", "A"), ("DE", "B")], names=["r_p", "s_p"])
    first, second = long_rows_mod.pair_labels(multi_index)
    assert first.tolist() == ["FR", "DE"]
    assert second.tolist() == ["A", "B"]

    tuple_index = pd.Index([("FR", "A"), ("DE",), "US"])
    first, second = long_rows_mod.pair_labels(tuple_index)
    assert first.tolist() == ["FR", "DE", "US"]
    assert second.tolist() == ["A", "", ""]

    one_level_multi = pd.MultiIndex.from_arrays([["FR", "DE"]], names=["r_p"])
    first, second = long_rows_mod.pair_labels(one_level_multi)
    assert first.tolist() == ["FR", "DE"]
    assert second.tolist() == ["", ""]

    impact_labels = np.asarray(["impact_a", "impact_b"], dtype=object)
    stage_r_labels = np.asarray(["FR", "DE"], dtype=object)
    stage_s_labels = np.asarray(["A", "B"], dtype=object)

    empty_stage = long_rows_mod.stage_rows_from_values(
        impact_labels=impact_labels,
        stage_r_labels=stage_r_labels,
        stage_s_labels=stage_s_labels,
        direct_values=np.zeros((2, 2)),
        embedded_values=np.zeros((2, 2)),
        total_values=np.zeros((2, 2)),
        eps=0.0,
    )
    assert list(empty_stage.columns) == [
        "impact",
        "stage_r_p",
        "stage_s_p",
        "direct_at_stage",
        "embedded_from_deeper_stages",
        "stage_total",
    ]

    stage_rows = long_rows_mod.stage_rows_from_values(
        impact_labels=impact_labels,
        stage_r_labels=stage_r_labels,
        stage_s_labels=stage_s_labels,
        direct_values=np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        embedded_values=np.asarray([[0.5, 0.0], [0.0, 0.0]]),
        total_values=np.asarray([[1.5, 0.0], [0.0, 0.0]]),
        eps=0.0,
    )
    assert stage_rows.to_dict("records") == [
        {
            "impact": "impact_a",
            "stage_r_p": "FR",
            "stage_s_p": "A",
            "direct_at_stage": 1.0,
            "embedded_from_deeper_stages": 0.5,
            "stage_total": 1.5,
        }
    ]

    empty_values = long_rows_mod.value_rows_from_values(
        impact_labels=impact_labels,
        r_labels=stage_r_labels,
        s_labels=stage_s_labels,
        values=np.zeros((2, 2)),
        eps=0.0,
        value_column="origin_value",
        r_column="origin_r_p",
        s_column="origin_s_p",
    )
    assert list(empty_values.columns) == ["impact", "origin_r_p", "origin_s_p", "origin_value"]

    value_rows = long_rows_mod.value_rows_from_values(
        impact_labels=impact_labels,
        r_labels=stage_r_labels,
        s_labels=stage_s_labels,
        values=np.asarray([[0.0, 2.0], [0.0, 0.0]]),
        eps=0.0,
        value_column="origin_value",
        r_column="origin_r_p",
        s_column="origin_s_p",
    )
    assert value_rows.to_dict("records") == [
        {
            "impact": "impact_a",
            "origin_r_p": "DE",
            "origin_s_p": "B",
            "origin_value": 2.0,
        }
    ]


def test_io_lca_column_definition_contracts_cover_year_and_duplicate_paths(
    tmp_path: Path,
) -> None:
    assert column_defs_mod._is_year_column("") is False  # noqa: SLF001
    assert column_defs_mod._is_year_column("2019.5") is False  # noqa: SLF001
    assert column_defs_mod._is_year_column("1700") is False  # noqa: SLF001
    assert column_defs_mod._is_year_column("2601") is False  # noqa: SLF001
    assert column_defs_mod._is_year_column("2019") is True  # noqa: SLF001
    assert column_defs_mod._selector_description("custom_selector").strip()  # noqa: SLF001

    origin_text = column_defs_mod.render_origin_columns_defs(
        columns=["impact", "impact", "custom_selector", "2019"]
    )
    origin_lines = origin_text.splitlines()
    assert origin_lines.count("impact") == 1
    assert origin_lines.count("custom_selector") == 1
    assert origin_lines.count("2019") == 1
    assert "2019.5" not in origin_lines

    stage_text = column_defs_mod.render_stage_columns_defs(
        columns=["year", "year", "custom_selector", "stage_total", "custom_selector"]
    )
    stage_lines = stage_text.splitlines()
    assert stage_lines.count("year") == 1
    assert stage_lines.count("custom_selector") == 1
    assert stage_lines.count("stage_total") == 1

    origin_path = column_defs_mod.write_origin_columns_defs(
        path=tmp_path / "defs" / "origin.txt",
        columns=["impact", "custom_selector", "2019"],
    )
    stage_path = column_defs_mod.write_stage_columns_defs(
        path=tmp_path / "defs" / "stage.txt",
        columns=["year", "custom_selector", "stage_total"],
    )
    assert origin_path.read_text(encoding="utf-8") == column_defs_mod.render_origin_columns_defs(
        columns=["impact", "custom_selector", "2019"]
    )
    assert stage_path.read_text(encoding="utf-8") == column_defs_mod.render_stage_columns_defs(
        columns=["year", "custom_selector", "stage_total"]
    )
    assert all(len(line) <= 100 for line in origin_text.splitlines())
    assert all(len(line) <= 100 for line in stage_text.splitlines())


def test_io_lca_metadata_contracts_cover_roundtrip_defaults_and_contract_errors(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "io_lca" / "metadata.json"
    missing_payload = metadata_mod.load_scope_manifest(
        path=metadata_path,
        function_name="deterministic_io_lca",
    )
    assert missing_payload["function"] == "deterministic_io_lca"
    assert missing_payload["arguments"] is None

    signature = {"project_name": "proj", "source": "exiobase_396_ixi"}

    resolved_key, scope = metadata_mod.get_scope(payload=missing_payload, signature=signature)
    assert scope is None
    direct_payload = {
        "function": "deterministic_io_lca",
        "arguments": signature,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "complete": True,
        "paths_written": [],
        "status": {"main": {}, "origin": {}, "stages": {}, "figures": {}},
        "identity_key": resolved_key,
    }
    existing_key, existing_scope = metadata_mod.get_scope(
        payload=direct_payload,
        signature=signature,
    )
    assert existing_key == resolved_key
    assert existing_scope is direct_payload

    created_scope = metadata_mod.ensure_scope(
        payload=missing_payload,
        key=resolved_key,
        signature=signature,
        function_name="deterministic_io_lca",
    )
    assert created_scope["function"] == "deterministic_io_lca"
    assert created_scope["status"] == {"main": {}, "origin": {}, "stages": {}, "figures": {}}
    metadata_mod.save_scope_manifest(path=metadata_path, payload=created_scope)
    saved_payload = metadata_mod.load_scope_manifest(
        path=metadata_path,
        function_name="deterministic_io_lca",
    )
    assert saved_payload["function"] == "deterministic_io_lca"
    assert saved_payload["arguments"] == signature

    canonical_scope = {
        "function": "deterministic_io_lca",
        "arguments": signature,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "complete": False,
        "paths_written": [],
        "status": {"main": {}, "origin": {}, "stages": {}, "figures": {}},
        "identity_key": resolved_key,
    }
    existing_scope = metadata_mod.ensure_scope(
        payload=canonical_scope,
        key=resolved_key,
        signature=signature,
        function_name="deterministic_io_lca",
    )
    assert existing_scope is canonical_scope

    assert metadata_mod.scope_complete_and_existing({"complete": False}) is False
    assert (
        metadata_mod.scope_complete_and_existing({"complete": True, "paths_written": []}) is False
    )
    assert (
        metadata_mod.scope_complete_and_existing(
            {"complete": True, "paths_written": [metadata_path]}
        )
        is True
    )

    assert metadata_mod.get_lcia_method_years(
        scope={"status": {"main": {"pb_lcia": {"years_done": [2019, "2020"]}}}},
        section="main",
        lcia_method="pb_lcia",
    ) == [2019, 2020]

    complete_signature = {
        "project_name": "proj",
        "source": "exiobase_396_ixi",
        "group_reg": False,
        "group_sec": False,
        "group_version": None,
        "fu_code": "L1.a",
        "aggreg_indices": False,
        "output_format": "csv",
        "years": [2019, 2020],
        "lcia_methods": ["pb_lcia", "gwp100_lcia"],
        "selectors": {"r_f": ["FR", "DE"]},
    }
    complete_scope = {"arguments": complete_signature}
    compatible = metadata_mod.compatible_scope(
        payload=complete_scope,
        project_name="proj",
        source="exiobase_396_ixi",
        group_reg=False,
        group_sec=False,
        group_version=None,
        fu_code="L1.a",
        aggreg_indices=False,
        output_format="csv",
        requested_years={2019},
        requested_methods={"pb_lcia"},
        requested_selectors={"r_f": ("DE", "FR"), "r_c": tuple(), "r_p": tuple(), "s_p": tuple()},
    )
    assert compatible is complete_scope
    assert metadata_mod.require_scope_signature(scope=complete_scope) is complete_signature

    mutable_scope: dict[str, Any] = {
        "status": {},
        "paths_written": ["a.csv"],
    }
    metadata_mod.set_lcia_method_years(
        scope=mutable_scope,
        section="main",
        lcia_method="pb_lcia",
        years_done=[2020, 2019, 2019],
        skipped_by_year={2021: "skip"},
    )
    metadata_mod.set_lcia_method_years(
        scope=mutable_scope,
        section="main",
        lcia_method="pb_lcia",
        years_done=[2020],
    )
    metadata_mod.merge_written_paths(
        scope=mutable_scope,
        paths=[Path("b.csv"), Path("a.csv")],
    )
    metadata_mod.set_lcia_method_years(
        scope=mutable_scope,
        section="origin",
        lcia_method="pb_lcia",
        years_done=[2018],
    )
    metadata_mod.set_scope_complete(scope=mutable_scope, complete=True)
    assert mutable_scope["complete"] is True
    assert mutable_scope["paths_written"] == ["a.csv", "b.csv"]

    metadata_mod.set_figure_paths(
        scope=mutable_scope,
        lcia_method="pb_lcia",
        figure_paths=[Path("figure_b.png"), Path("figure_a.png"), Path("figure_a.png")],
    )
    assert mutable_scope["status"]["figures"]["pb_lcia"]["paths"] == [
        "figure_a.png",
        "figure_b.png",
    ]


def test_io_lca_deterministic_path_probes_do_not_create_optional_directories(
    tmp_path: Path,
) -> None:
    paths = paths_mod.IOLCAPaths(
        project_base=tmp_path / "proj",
        lca_root=tmp_path / "proj" / "A_lca" / "io_lca",
        source_version_token="original_version",
    )
    source = "exiobase_396_ixi"

    result_paths = [
        paths_mod.lca_results_dir_for_source(paths=paths, source=source),
        paths_mod.origin_dir_for_source(paths=paths, source=source),
        paths_mod.stages_dir_for_source(paths=paths, source=source),
        paths_mod.log_dir_for_source(paths=paths, source=source),
        paths_mod.figures_dir_for_source(paths=paths, source=source),
        paths_mod.io_metadata_path_for_source(paths=paths, source=source),
        paths_mod.figure_metadata_path_for_source(paths=paths, source=source),
        paths_mod.origin_columns_defs_path(paths=paths, source=source),
        paths_mod.stage_columns_defs_path(paths=paths, source=source),
        paths_mod.main_results_path(
            paths=paths,
            source=source,
            lcia_method="pb_lcia",
            extension="csv",
        ),
        paths_mod.origin_results_path(
            paths=paths,
            source=source,
            lcia_method="pb_lcia",
            extension="csv",
        ),
        paths_mod.origin_ratio_results_path(
            paths=paths,
            source=source,
            lcia_method="pb_lcia",
            extension="csv",
        ),
        paths_mod.stage_results_path(
            paths=paths,
            source=source,
            lcia_method="pb_lcia",
            year=2019,
            extension="csv",
        ),
    ]

    assert result_paths
    assert not paths_mod.source_scope_root_for_source(paths=paths, source=source).exists()

    base_table = tmp_path / "main.csv"
    expected = paths_mod.io_lca_expected_method_table_paths(
        base_path=base_table,
        lcia_methods=["gwp100_lcia", "pb_lcia"],
    )
    assert expected == [
        paths_mod.io_lca_method_table_path(base_path=base_table, lcia_method="gwp100_lcia"),
        paths_mod.io_lca_method_table_path(base_path=base_table, lcia_method="pb_lcia"),
    ]
    expected[0].parent.mkdir(parents=True, exist_ok=True)
    expected[0].write_text("x\n", encoding="utf-8")
    assert paths_mod.io_lca_method_table_paths(base_path=base_table) == [expected[0]]
    assert (
        paths_mod.io_lca_lcia_method_from_path(
            path=expected[0],
            file_stem=base_table.stem,
        )
        == "gwp100_lcia"
    )
