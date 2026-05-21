from pathlib import Path
from typing import Any

import pytest

from pyaesa.asocc.io.metadata import _build_run_metadata, _save_run_metadata
from pyaesa.asocc.orchestration.setup.reuse.completed_run_policy import (
    apply_completed_run_policy,
)
from pyaesa.asocc.orchestration.setup.request.types import _YearBundle
from pyaesa.asocc.runtime.paths.deterministic import _get_allocate_run_metadata_path
from pyaesa.asocc.runtime.paths.published import _asocc_deterministic_scope_root


def _signature() -> dict[str, object]:
    return {
        "source": "oecd_v2025",
        "group_version": None,
        "group_reg": False,
        "group_sec": False,
        "aggreg_indices": False,
        "l1_reg_aggreg": "post",
        "fu_code": "L1.a",
        "studied_indices_tag": "r_p-FR",
        "years": [2020, 2021],
        "lcia_methods": [],
        "ssp_scenario_input": None,
        "reference_years_input": None,
        "selected_methods": {"l1": ["EG(Pop)"], "l2_in_l1": [], "l2_vs_global": []},
        "projection_mode": None,
        "reg_window": None,
        "l2_reuse_years": [],
        "output_format": "csv",
        "intermediate_outputs": False,
    }


def _year_bundle() -> _YearBundle:
    return _YearBundle(
        resolved_years=[2020, 2021],
        historical_years=[2020, 2021],
        max_year=2021,
        out_of_range_years=[],
    )


def _metadata_path(tmp_path: Path, signature: dict[str, object]) -> Path:
    return _get_allocate_run_metadata_path(
        tmp_path,
        source=str(signature["source"]),
        group_version=None,
    )


def _output_file(tmp_path: Path, signature: dict[str, object]) -> Path:
    root = _asocc_deterministic_scope_root(
        proj_base=tmp_path,
        source=str(signature["source"]),
        group_version=None,
    )
    return root / "results" / "table_l1.csv"


def _write_metadata(
    tmp_path: Path,
    signature: dict[str, object],
    scope_payload: dict[str, Any],
) -> None:
    path = _metadata_path(tmp_path, signature)
    raw_scope = dict(scope_payload)
    scope_signature = dict(raw_scope.get("signature", signature))
    persisted = _build_run_metadata(
        requested_years=list(scope_signature.get("years", [])),
        resolved_years=list(raw_scope.get("completed_years", [])),
        selected_methods=dict(scope_signature.get("selected_methods", {})),
        fu_code=str(scope_signature["fu_code"]),
        studied_indices_tag=str(scope_signature.get("studied_indices_tag", "")),
        skipped_years={},
        outputs=[str(path) for path in raw_scope.get("outputs", [])],
        signature=scope_signature,
    )
    persisted["execution"]["completed_years"] = list(raw_scope.get("completed_years", []))
    persisted["provenance"]["ssp_scenarios"] = raw_scope.get("ssp_scenarios")
    _save_run_metadata(path, persisted)


def _apply(
    tmp_path: Path,
    signature: dict[str, object],
    *,
    refresh: bool = False,
    requested_years: list[int] | None = None,
):
    return apply_completed_run_policy(
        refresh=refresh,
        proj_base=tmp_path,
        output_source=str(signature["source"]),
        run_signature=signature,
        year_bundle=_year_bundle(),
        reference_years=None,
        requested_years=requested_years or [2020, 2021],
        ssp_scenario_options=[],
    )


def test_completed_run_policy_skips_exact_complete_scope(tmp_path: Path) -> None:
    signature = _signature()
    output = _output_file(tmp_path, signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        signature,
        {
            "signature": signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    (
        year_bundle,
        reference_years,
        scenarios,
        is_complete,
        metadata_years,
        prior_outputs,
        append_scope,
    ) = _apply(tmp_path, signature)

    assert year_bundle == _year_bundle()
    assert reference_years is None
    assert scenarios == []
    assert is_complete is True
    assert metadata_years == [2020, 2021]
    assert prior_outputs == [str(output)]
    assert append_scope is None


def test_completed_run_policy_recomputes_fresh_scope_and_refresh(tmp_path: Path) -> None:
    signature = _signature()

    assert _apply(tmp_path, signature)[3] is False

    output = _output_file(tmp_path, signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    assert _apply(tmp_path, signature, refresh=True)[3] is False


def test_completed_run_policy_rejects_outputs_without_metadata(tmp_path: Path) -> None:
    signature = _signature()
    output = _output_file(tmp_path, signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _apply(tmp_path, signature)


def test_completed_run_policy_rejects_different_fu_existing_metadata(tmp_path: Path) -> None:
    signature = _signature()
    other_signature = {**signature, "source": "exiobase_3102_ixi", "fu_code": "L1.b"}
    _write_metadata(
        tmp_path,
        other_signature,
        {
            "signature": other_signature,
            "completed_years": [2020, 2021],
            "outputs": ["some.csv"],
        },
    )

    with pytest.raises(ValueError):
        _apply(tmp_path, signature)


def test_completed_run_policy_rejects_different_format_metadata(
    tmp_path: Path,
) -> None:
    signature = _signature()
    other_signature = {**signature, "output_format": "parquet"}
    _write_metadata(
        tmp_path,
        other_signature,
        {
            "signature": other_signature,
            "completed_years": [2020, 2021],
            "outputs": ["some.parquet"],
        },
    )

    with pytest.raises(ValueError):
        _apply(tmp_path, signature)


def test_completed_run_policy_rejects_scope_with_missing_outputs(tmp_path: Path) -> None:
    signature = _signature()
    missing_output = _output_file(tmp_path, signature)
    _write_metadata(
        tmp_path,
        signature,
        {"completed_years": [2020, 2021], "outputs": [str(missing_output)]},
    )

    with pytest.raises(ValueError):
        _apply(tmp_path, signature)


def test_completed_run_policy_allows_year_append_for_incomplete_scope(tmp_path: Path) -> None:
    signature = _signature()
    output = _output_file(tmp_path, signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        signature,
        {"completed_years": [2020], "outputs": [str(output)]},
    )

    assert _apply(tmp_path, signature)[3] is False


def test_completed_run_policy_reuses_subset_scope(tmp_path: Path) -> None:
    persisted_signature = {
        **_signature(),
        "years": [2020, 2021, 2022],
        "ssp_scenario_input": "SSP1",
    }
    requested_signature = {**persisted_signature, "years": [2020, 2021]}
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021,2022\nEG(Pop),0.5,0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021, 2022],
            "outputs": [str(output)],
        },
    )

    assert _apply(tmp_path, requested_signature, requested_years=[2020, 2021])[3] is True


def test_completed_run_policy_allows_compatible_superset_scope(tmp_path: Path) -> None:
    persisted_signature = _signature()
    requested_signature = {**persisted_signature, "years": [2020, 2021, 2022]}
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    result = _apply(tmp_path, requested_signature, requested_years=[2020, 2021, 2022])

    assert result[0].resolved_years == [2022]
    assert result[3] is False
    assert result[4] == [2020, 2021, 2022]
    assert result[5] == [str(output)]
    assert result[6] is not None
    assert result[6].years == [2022]


def test_completed_run_policy_prunes_scalar_selector_append_scope(tmp_path: Path) -> None:
    persisted_signature = {**_signature(), "ssp_scenario_input": None}
    requested_signature = {**persisted_signature, "ssp_scenario_input": "SSP1"}
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    result = _apply(tmp_path, requested_signature)

    assert result[0].resolved_years == [2020, 2021]
    assert result[3] is False
    assert result[6] is not None
    assert result[6].ssp_scenario_input == ["SSP1"]


def test_completed_run_policy_prunes_selected_method_append_scope(tmp_path: Path) -> None:
    persisted_signature = {
        **_signature(),
        "selected_methods": {"l1": ["EG(Pop)"], "l2_in_l1": [], "l2_vs_global": []},
    }
    requested_signature = {
        **persisted_signature,
        "selected_methods": {
            "l1": ["EG(Pop)", "PR(GDPcap)"],
            "l2_in_l1": [],
            "l2_vs_global": [],
        },
    }
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    result = _apply(tmp_path, requested_signature)

    assert result[0].resolved_years == [2020, 2021]
    assert result[3] is False
    assert result[6] is not None
    assert result[6].selected_methods == {
        "l1": ["PR(GDPcap)"],
        "l2_in_l1": [],
        "l2_vs_global": [],
    }


def test_completed_run_policy_rejects_mixed_subset_superset_scope(tmp_path: Path) -> None:
    persisted_signature = {
        **_signature(),
        "years": [2020, 2021, 2022],
        "lcia_methods": ["gwp100_lcia"],
    }
    requested_signature = {
        **persisted_signature,
        "years": [2020, 2021],
        "lcia_methods": ["gwp100_lcia", "pb_lcia"],
    }
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021,2022\nEG(Pop),0.5,0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021, 2022],
            "outputs": [str(output)],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        _apply(tmp_path, requested_signature, requested_years=[2020, 2021])
    message = str(exc_info.value)
    assert "years" in message
    assert "lcia_methods" in message
    assert "pb_lcia" in message
    assert "2022" in message


def test_completed_run_policy_rejects_partial_axis_overlap(tmp_path: Path) -> None:
    persisted_signature = {
        **_signature(),
        "lcia_methods": ["gwp100_lcia", "water_lcia"],
    }
    requested_signature = {
        **persisted_signature,
        "lcia_methods": ["gwp100_lcia", "pb_lcia"],
    }
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        _apply(tmp_path, requested_signature)
    message = str(exc_info.value)
    assert "lcia_methods" in message
    assert "pb_lcia" in message
    assert "water_lcia" in message


def test_completed_run_policy_rejects_studied_filter_change(tmp_path: Path) -> None:
    persisted_signature = _signature()
    requested_signature = {**persisted_signature, "studied_indices_tag": "r_p-US"}
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    with pytest.raises(ValueError):
        _apply(tmp_path, requested_signature)


def test_completed_run_policy_rejects_regression_window_change(tmp_path: Path) -> None:
    persisted_signature = {
        **_signature(),
        "projection_mode": "regression",
        "reg_window": [2005, 2019],
    }
    requested_signature = {**persisted_signature, "reg_window": [2006, 2019]}
    output = _output_file(tmp_path, persisted_signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("method,2020,2021\nEG(Pop),0.5,0.5\n", encoding="utf-8")
    _write_metadata(
        tmp_path,
        persisted_signature,
        {
            "signature": persisted_signature,
            "completed_years": [2020, 2021],
            "outputs": [str(output)],
        },
    )

    with pytest.raises(ValueError):
        _apply(tmp_path, requested_signature)
    assert _apply(tmp_path, {**persisted_signature, "reg_window": None})[3] is True
