from pathlib import Path

import pytest

from pyaesa.shared.runtime.io import persisted_paths
from pyaesa.shared.runtime.reuse import branch_reuse
from pyaesa.shared.runtime.reuse import contracts
from pyaesa.shared.runtime.reuse import derived_state
from pyaesa.shared.runtime.scenario import file_routing as scenario_file_routing
from pyaesa.shared.runtime.text import (
    compact_user_text,
    extend_user_text_lines,
    join_user_text_lines,
    print_user_text_line,
    wrap_user_text_lines,
)


def test_user_text_wrapping_preserves_bullets_and_blank_lines() -> None:
    lines = [
        "Heading",
        "",
        "- This bullet is intentionally long enough to wrap across more than one line "
        "while keeping the continuation aligned under the bullet text.",
        "This plain line is also intentionally long enough to wrap without bullet "
        "indentation while preserving the words in their original order.",
    ]

    wrapped = wrap_user_text_lines(lines, width=72)
    assert "" in wrapped
    assert any(line.startswith("- This bullet") for line in wrapped)
    assert any(line.startswith("  line while keeping") for line in wrapped)
    assert all(len(line) <= 72 for line in wrapped)

    text = join_user_text_lines(lines, width=72, trailing_newline=True)
    assert text.endswith("\n")
    assert " ".join(text.split()).startswith("Heading - This bullet")


def test_user_text_helpers_extend_print_and_compact(capsys) -> None:
    lines = ["Header"]
    extend_user_text_lines(
        lines,
        "  Active uncertainty sources: projection_uncertainty, reference_year_uncertainty",
        width=48,
    )
    assert all(len(line) <= 48 for line in lines)
    assert lines[1].startswith("  Active uncertainty sources:")

    print_user_text_line("This message is deliberately long enough to wrap once.", width=36)
    captured = capsys.readouterr()
    assert all(len(line) <= 36 for line in captured.out.splitlines())

    assert compact_user_text("  A   compact   label  ", max_chars=100) == "A compact label"
    assert compact_user_text("abcdef", max_chars=5) == "ab..."
    assert compact_user_text("abcdef", max_chars=2) == "ab"


def test_normalize_and_scope_persisted_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    csv_path = root / "a.csv"
    parquet_path = root / "b.parquet"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    parquet_path.write_text("stub", encoding="utf-8")

    assert persisted_paths.normalize_persisted_paths(raw_paths=None) == []
    assert persisted_paths.normalize_persisted_paths(
        raw_paths=[csv_path, f"  {parquet_path}  "],
    ) == [csv_path, parquet_path]

    assert persisted_paths.scoped_existing_table_paths(
        raw_paths=[parquet_path, csv_path],
        root=root,
        field_name="paths",
    ) == sorted([csv_path.resolve(), parquet_path.resolve()])

    outside = tmp_path / "outside.csv"
    outside.write_text("x\n2\n", encoding="utf-8")
    bad_suffix = root / "bad.txt"
    bad_suffix.write_text("bad", encoding="utf-8")
    missing = root / "missing.csv"
    with pytest.raises(ValueError):
        persisted_paths.scoped_existing_table_paths(
            raw_paths=[outside],
            root=root,
            field_name="paths",
        )
    with pytest.raises(ValueError):
        persisted_paths.scoped_existing_table_paths(
            raw_paths=[bad_suffix],
            root=root,
            field_name="paths",
        )
    with pytest.raises(ValueError):
        persisted_paths.scoped_existing_table_paths(
            raw_paths=[missing],
            root=root,
            field_name="paths",
        )
    with pytest.raises(ValueError):
        persisted_paths.scoped_existing_table_paths(
            raw_paths=[csv_path, csv_path],
            root=root,
            field_name="paths",
        )


def test_derived_state_request_matching_and_persistence(tmp_path: Path) -> None:
    payload: dict[str, object] = {}
    state_key = "figure_state"
    request_signature = {"dpi": 150}
    compute_signature = {"year": 2030}
    path = tmp_path / "figure.png"
    path.write_text("png", encoding="utf-8")

    assert derived_state._state_block(payload, state_key=state_key) is None

    assert derived_state._paths_exist([]) is False
    assert derived_state._paths_exist([path]) is True
    path.unlink()
    assert derived_state._paths_exist([path]) is False
    path.write_text("png", encoding="utf-8")

    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature=request_signature,
        )
        is False
    )

    derived_state.set_request_state(
        payload=payload,
        state_key=state_key,
        request_signature=request_signature,
        compute_signature=compute_signature,
        paths=[path, path],
        extra={"note": "kept"},
    )
    block = derived_state._state_block(payload, state_key=state_key)
    assert block is not None
    assert block["note"] == "kept"
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature=request_signature,
            compute_signature=compute_signature,
        )
        is True
    )
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature={"dpi": 10},
            compute_signature={"year": 2030},
            request_compatible=lambda stored, requested: stored["dpi"] >= requested["dpi"],
            compute_compatible=lambda stored, requested: stored["year"] == requested["year"],
        )
        is True
    )
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature={"dpi": 10},
            compute_signature={"year": 2040},
            request_compatible=lambda stored, requested: stored["dpi"] >= requested["dpi"],
            compute_compatible=lambda stored, requested: stored["year"] == requested["year"],
        )
        is False
    )
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature={"dpi": 300},
            compute_signature={"year": 2030},
            request_compatible=lambda stored, requested: stored["dpi"] >= requested["dpi"],
        )
        is False
    )
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature={"dpi": 300},
            compute_signature=compute_signature,
        )
        is False
    )
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature=request_signature,
            compute_signature={"year": 2005},
        )
        is False
    )
    assert (
        derived_state.request_state_matches(
            payload={state_key: {"request_signature": "bad", "paths": [str(path)]}},
            state_key=state_key,
            request_signature=request_signature,
            request_compatible=lambda stored, requested: stored == requested,
        )
        is False
    )
    path.unlink()
    assert (
        derived_state.request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature=request_signature,
            compute_signature=compute_signature,
        )
        is False
    )

    payload_without_compute: dict[str, object] = {}
    derived_state.set_request_state(
        payload=payload_without_compute,
        state_key=state_key,
        request_signature=request_signature,
        paths=[path],
    )
    state_payload = payload_without_compute[state_key]
    assert isinstance(state_payload, dict)
    assert "compute_signature" not in state_payload


def test_asocc_signature_reuse_falls_back_to_scope_ssp_scenarios() -> None:
    class _Signature:
        def as_dict(self) -> dict:
            return {
                "source": "oecd_v2025",
                "agg_version": None,
                "agg_reg": False,
                "agg_sec": False,
                "fu_code": "L2.a.a",
                "studied_indices_tag": "demo",
                "l1_reg_aggreg": "post",
                "variant_tag": None,
                "group_indices": False,
                "projection_mode": None,
                "reg_window": None,
                "selected_methods": {"L2": ["UT(FD)"]},
                "ssp_scenario_input": None,
            }

    class _Scope:
        compute_signature = _Signature()
        ssp_scenarios = ["SSP2"]

    requested = {
        "source": "oecd_v2025",
        "agg_version": None,
        "agg_reg": False,
        "agg_sec": False,
        "fu_code": "L2.a.a",
        "studied_indices_tag": "demo",
        "l1_reg_aggreg": "post",
        "variant_tag": None,
        "group_indices": False,
        "projection_mode": None,
        "reg_window": None,
        "selected_methods": {"L2": ["UT(FD)"]},
        "ssp_scenario_input": ["SSP2"],
    }

    assert contracts.asocc_signature_matches_request(
        requested_signature=requested,
        scope=_Scope(),
        run_ssp_scenarios=None,
    )
    assert contracts.asocc_signature_matches_request(
        requested_signature={**requested, "ssp_scenario_input": ["SSP5"]},
        scope=_Scope(),
        run_ssp_scenarios=None,
        check_ssp=False,
    )


def test_branch_reuse_contracts_cover_refresh_cleanup(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "table.csv"
    output_path.write_text("x\n1\n", encoding="utf-8")
    missing_output_path = tmp_path / "missing.csv"
    branch_reuse.cleanup_branch_outputs_for_refresh(
        existing_metadata={
            "artifacts": {"result_paths": [str(output_path), str(missing_output_path)]}
        },
        meta_path=meta_path,
        artifact_keys=("result_paths",),
    )
    assert output_path.exists() is False
    assert meta_path.exists() is False

    branch_reuse.cleanup_branch_outputs_for_refresh(
        existing_metadata={"artifacts": {"result_paths": []}},
        meta_path=meta_path,
        artifact_keys=("result_paths",),
    )
    assert meta_path.exists() is False

    scoped_root = tmp_path / "scoped_root"
    nested_file = scoped_root / "nested" / "leftover.csv"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("x\n1\n", encoding="utf-8")
    branch_reuse.cleanup_branch_outputs_for_refresh(
        existing_metadata={"artifacts": {"result_paths": []}},
        meta_path=meta_path,
        artifact_keys=("result_paths",),
        scope_targets=(scoped_root,),
    )
    assert scoped_root.exists() is False

    scoped_dir = tmp_path / "scoped"
    scoped_dir.mkdir()
    scoped_file = scoped_dir / "kept.csv"
    scoped_file.write_text("x\n1\n", encoding="utf-8")
    loose_file = tmp_path / "loose.json"
    loose_file.write_text("{}", encoding="utf-8")
    branch_reuse.cleanup_refresh_scope_targets(
        targets=(scoped_dir, loose_file, tmp_path / "missing"),
    )
    assert scoped_dir.exists() is False
    assert loose_file.exists() is False


def test_scenario_file_routing_validation_and_resolution(tmp_path: Path) -> None:
    hist = scenario_file_routing.ScenarioTaggedFileSpec(
        path=tmp_path / "hist.csv",
        scenario=None,
        years=(2005, 2006),
    )
    ssp2 = scenario_file_routing.ScenarioTaggedFileSpec(
        path=tmp_path / "ssp2.csv",
        scenario="SSP2",
        years=(2030, 2040),
    )
    ssp1 = scenario_file_routing.ScenarioTaggedFileSpec(
        path=tmp_path / "ssp1.csv",
        scenario="SSP1",
        years=(2030, 2040),
    )

    assert (
        scenario_file_routing.allowed_scenarios_for_year(
            year=2030,
            ssp_scenario_options_by_year=None,
        )
        == set()
    )
    assert scenario_file_routing.allowed_scenarios_for_year(
        year=2030,
        ssp_scenario_options_by_year={2030: [None, "SSP2"]},
    ) == {"SSP2"}

    scenario_file_routing.validate_scenario_inventory(
        specs=(),
        family_label="external files",
        item_label="numerator",
    )
    scenario_file_routing.validate_scenario_inventory(
        specs=(hist, ssp2),
        family_label="external files",
        item_label="numerator",
    )

    with pytest.raises(ValueError):
        scenario_file_routing.validate_scenario_inventory(
            specs=(ssp2, ssp2),
            family_label="external files",
            item_label="numerator",
        )
    with pytest.raises(ValueError):
        scenario_file_routing.validate_scenario_inventory(
            specs=(
                ssp2,
                scenario_file_routing.ScenarioTaggedFileSpec(
                    path=tmp_path / "ssp3.csv",
                    scenario="SSP3",
                    years=(2030,),
                ),
            ),
            family_label="external files",
            item_label="numerator",
        )
    with pytest.raises(ValueError):
        scenario_file_routing.validate_scenario_inventory(
            specs=(
                hist,
                scenario_file_routing.ScenarioTaggedFileSpec(
                    path=tmp_path / "overlap.csv",
                    scenario="SSP2",
                    years=(2006, 2030),
                ),
            ),
            family_label="external files",
            item_label="numerator",
        )

    assignments = scenario_file_routing.resolve_year_assignments(
        specs=(hist, ssp1, ssp2),
        years=[2005, 2030],
        ssp_scenario_options_by_year={2030: ["SSP2"]},
        family_label="external files",
        item_label="numerator",
        expected_stems=["hist", "ssp2"],
    )
    assert assignments == {hist.path: [2005], ssp2.path: [2030]}

    multi_assignments = scenario_file_routing.resolve_year_assignments(
        specs=(ssp1, ssp2),
        years=[2030],
        ssp_scenario_options_by_year={2030: ["SSP1", "SSP2"]},
        family_label="external files",
        item_label="numerator",
        expected_stems=["ssp1", "ssp2"],
    )
    assert multi_assignments == {ssp1.path: [2030], ssp2.path: [2030]}

    with pytest.raises(ValueError):
        scenario_file_routing.resolve_year_assignments(
            specs=(hist,),
            years=[1995],
            ssp_scenario_options_by_year=None,
            family_label="external files",
            item_label="numerator",
        )
    with pytest.raises(ValueError):
        scenario_file_routing.resolve_year_assignments(
            specs=(ssp1, ssp2),
            years=[2030],
            ssp_scenario_options_by_year={2030: ["SSP9"]},
            family_label="external files",
            item_label="numerator",
        )
