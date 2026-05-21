"""Tests for runtime environment metadata in MRIO year entries."""

from pyaesa.process.mrios.utils.pipeline.lcia_tracking import (
    build_year_entry_payload,
)


def test_build_year_entry_payload_includes_runtime_env() -> None:
    """Year entry should persist runtime environment versions."""
    runtime_env = {
        "python": "3.11.9",
        "numpy": "1.26.4",
        "pandas": "2.3.3",
    }
    payload = build_year_entry_payload(
        saved_dir_name="IOT_2019_ixi_calc",
        core_matrices=["A", "L", "Y"],
        extension_payload={},
        updated_iso="2026-02-28T00:00:00+00:00",
        uncasext_only=True,
        preclip_core_matrices=[],
        preclip_extension_payload={},
        pymrio_calc_all=False,
        enacting_metric_units={"mrio_default_monetary": "M.EUR"},
        applied_methods=[],
        is_exio=True,
        requires_characterization=False,
        year_char_jobs={},
        missing_by_method=None,
        runtime_env=runtime_env,
    )

    assert payload["runtime_env"] == runtime_env
    assert "raw_corrected_values" not in payload


def test_build_year_entry_payload_includes_raw_corrected_values_payload() -> None:
    payload = build_year_entry_payload(
        saved_dir_name="IOT_2020_ixi_calc",
        core_matrices=["A"],
        extension_payload={},
        updated_iso="2026-02-28T00:00:00+00:00",
        uncasext_only=True,
        preclip_core_matrices=[],
        preclip_extension_payload={},
        pymrio_calc_all=False,
        enacting_metric_units={"mrio_default_monetary": "M.EUR"},
        applied_methods=[],
        is_exio=True,
        requires_characterization=False,
        year_char_jobs={},
        missing_by_method=None,
        runtime_env={"python": "3.11.9"},
        raw_correction_payload={
            "row_count": 3,
            "log_path": "logs/raw_corrected_values.csv",
            "summary_lines": ["2020: CH nutrients/P - agriculture - water sectors corrected."],
        },
    )
    assert payload["raw_corrected_values"] == {
        "row_count": 3,
        "log_path": "logs/raw_corrected_values.csv",
        "summary_lines": ["2020: CH nutrients/P - agriculture - water sectors corrected."],
    }
