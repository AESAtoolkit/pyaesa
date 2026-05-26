"""Extra strict failure coverage for disaggregation config parsing."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pyaesa.asocc.disaggregation.branch_context import validate_region_compatibility
from pyaesa.asocc.disaggregation.config import parse_disaggregate_args
from pyaesa.asocc.disaggregation.models import DisaggregationBranchReport, DisaggregationReport
from pyaesa.asocc.disaggregation.run_plan import build_disaggregation_run_plan


def _base_config() -> dict:
    return {
        "target_agg_run": {
            "source": "oecd_v2025",
            "agg_reg": False,
            "agg_sec": False,
            "agg_version": None,
            "s_p": ["D"],
        },
        "ref_agg_run": {
            "source": "exiobase_396_ixi",
            "agg_reg": False,
            "agg_sec": True,
            "agg_version": "oecd_d",
            "s_p": ["D"],
        },
        "ref_disagg_run": {
            "source": "exiobase_396_ixi",
            "agg_reg": False,
            "agg_sec": True,
            "agg_version": "elec",
            "s_p": ["Electricity"],
        },
        "disaggregation_specs": [{"agg_sector_label": "D", "disagg_sector_label": "Electricity"}],
        "new_disagg_version_name": "disagg_oecd",
    }


def _base_allocate_args() -> dict:
    return {"project_name": "p", "fu_code": "L2.c.b"}


def _call(config: dict, args: dict, **runtime):
    defaults = {
        "output_format": "csv",
        "figures": False,
        "figure_options": None,
        "figure_format": None,
        "figure_external_method": None,
        "refresh": False,
    }
    defaults.update(runtime)
    return parse_disaggregate_args(
        disaggregation_config=config,
        base_allocate_args=args,
        output_format=defaults["output_format"],
        figures=defaults["figures"],
        figure_options=defaults["figure_options"],
        figure_format=defaults["figure_format"],
        figure_external_method=defaults["figure_external_method"],
        refresh=defaults["refresh"],
    )


def test_parse_disaggregate_args_rejects_non_dict_config() -> None:
    with pytest.raises(ValueError, match="disaggregation_config"):
        parse_disaggregate_args(
            disaggregation_config=cast(Any, "bad"),
            base_allocate_args=_base_allocate_args(),
            figures=False,
            figure_options=None,
            figure_format=None,
            figure_external_method=None,
            output_format="csv",
            refresh=False,
        )


def test_parse_disaggregate_args_rejects_missing_and_unknown_top_keys() -> None:
    config = _base_config()
    del config["new_disagg_version_name"]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["extra"] = 1
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())


def test_parse_selector_strict_failure_modes() -> None:
    config = _base_config()
    config["target_agg_run"] = "bad"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["unknown"] = 1
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["source"] = ""
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["source"] = None
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["source"] = "bad_source"
    with pytest.raises(ValueError, match="source"):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["agg_reg"] = "False"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["agg_version"] = "x"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["s_p"] = "D"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["s_p"] = [""]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["s_p"] = [None]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["target_agg_run"]["s_p"] = []
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())


def test_parse_specs_strict_failure_modes() -> None:
    config = _base_config()
    config["disaggregation_specs"] = "bad"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["disaggregation_specs"] = [1]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["disaggregation_specs"] = [
        {
            "agg_sector_label": "D",
            "disagg_sector_label": "Electricity",
            "extra": "x",
        }
    ]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["disaggregation_specs"] = [{"agg_sector_label": "D", "disagg_sector_label": ""}]
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())


def test_selector_to_specs_cross_rules_failures() -> None:
    config = _base_config()
    config["ref_agg_run"]["s_p"] = ["X"]
    with pytest.raises(ValueError, match="ref_agg_run.s_p"):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["ref_disagg_run"]["s_p"] = ["X"]
    with pytest.raises(ValueError, match="ref_disagg_run.s_p"):
        _call(config, _base_allocate_args())

    config = _base_config()
    config["ref_disagg_run"]["source"] = "oecd_v2025"
    with pytest.raises(ValueError):
        _call(config, _base_allocate_args())


def test_base_allocate_args_strict_failure_modes() -> None:
    with pytest.raises(ValueError):
        _call(_base_config(), "bad")  # type: ignore[arg type]

    args = _base_allocate_args()
    args["extra"] = 1
    with pytest.raises(ValueError):
        _call(_base_config(), args)

    args = _base_allocate_args()
    args["source"] = "oecd_v2025"
    with pytest.raises(ValueError):
        _call(_base_config(), args)

    args = {"fu_code": "L2.c.b"}
    with pytest.raises(ValueError, match="project_name"):
        _call(_base_config(), args)

    args = {"project_name": "p"}
    with pytest.raises(ValueError):
        _call(_base_config(), args)

    args = {"project_name": "p", "fu_code": "L1.a"}
    with pytest.raises(ValueError):
        _call(_base_config(), args)


def test_runtime_type_failures() -> None:
    with pytest.raises(ValueError, match="refresh"):
        _call(_base_config(), _base_allocate_args(), refresh=1)


def test_agg_flags_default_to_false_when_omitted() -> None:
    config = _base_config()
    del config["target_agg_run"]["agg_reg"]
    del config["target_agg_run"]["agg_sec"]
    parsed = _call(config, _base_allocate_args())
    assert parsed.disaggregation.target_agg_run.agg_reg is False
    assert parsed.disaggregation.target_agg_run.agg_sec is False


def test_base_allocate_time_selector_normalization_and_failures() -> None:
    parsed = _call(
        _base_config(),
        {
            "project_name": "p",
            "fu_code": "L2.c.b",
            "years": range(2005, 2007),
            "l2_reuse_years": [2012, 2013],
            "reg_window": range(2001, 2004),
        },
    )
    assert list(parsed.base_allocate_args["years"]) == [2005, 2006]
    assert parsed.base_allocate_args["l2_reuse_years"] == [2012, 2013]
    assert list(parsed.base_allocate_args["reg_window"]) == [2001, 2002, 2003]

    args = _base_allocate_args()
    args["years"] = (2005, 2006)
    with pytest.raises(ValueError, match="years"):
        _call(_base_config(), args)

    args = _base_allocate_args()
    args["reg_window"] = (2001, 2003)
    with pytest.raises(ValueError, match="reg_window"):
        _call(_base_config(), args)

    args = _base_allocate_args()
    args["reg_window"] = "bad"
    with pytest.raises(ValueError, match="reg_window"):
        _call(_base_config(), args)


def test_disaggregation_run_plan_rejects_invalid_filters_and_empty_method_selection() -> None:
    parsed = _call(_base_config(), {**_base_allocate_args(), "r_p": 1})
    with pytest.raises(ValueError, match="base_asocc_args.r_p"):
        build_disaggregation_run_plan(parsed)


def test_region_compatibility_rejects_invalid_region_filters_and_missing_labels(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    selector = SimpleNamespace(
        source="oecd_v2025",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    )
    base_args = {
        "r_p": None,
        "r_c": "FR",
        "r_f": ["US"],
    }
    validate_region_compatibility(
        target_selector=selector,
        ref_aggregated_selector=selector,
        ref_disaggregate_selector=selector,
        base_allocate_args=base_args,
        combined_methods=[],
    )

    with pytest.raises(ValueError):
        validate_region_compatibility(
            target_selector=selector,
            ref_aggregated_selector=selector,
            ref_disaggregate_selector=selector,
            base_allocate_args={"r_p": ["ZZ"], "r_c": None, "r_f": None},
            combined_methods=[],
        )

    parsed = _call(
        _base_config(),
        {
            **_base_allocate_args(),
            "method_plan": "one_step",
            "one_step_methods": ["UT(TD)"],
            "r_p": ["FR"],
            "r_c": ["FR"],
        },
    )
    with pytest.raises(ValueError):
        build_disaggregation_run_plan(parsed)

    parsed = _call(
        _base_config(),
        {
            **_base_allocate_args(),
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_TD})"],
        },
    )
    with pytest.raises(ValueError):
        build_disaggregation_run_plan(parsed)


def test_disaggregation_report_string_covers_figure_summaries() -> None:
    with_figures = DisaggregationReport(
        source_label="demo",
        branch_reports=[
            DisaggregationBranchReport(
                l1_reg_aggreg="post",
                group_indices=True,
                summaries=["done"],
                disaggregation_audit_path=Path("audit.csv"),
                metadata_path=Path("metadata.json"),
                figure_paths=[Path("figures/out.png")],
            )
        ],
    )
    with_figure_summary = str(with_figures)
    assert "group_indices=grouped" in with_figure_summary
    assert "done" in with_figure_summary

    plain = DisaggregationReport(
        source_label="demo",
        branch_reports=[
            DisaggregationBranchReport(
                l1_reg_aggreg="post",
                group_indices=False,
                summaries=["done"],
                disaggregation_audit_path=Path("audit.csv"),
                metadata_path=Path("metadata.json"),
                figure_paths=[],
            )
        ],
    )
    plain_summary = str(plain)
    assert plain_summary
    assert plain.branch_reports[0].figure_paths == []


def test_parse_disaggregate_args_rejects_figure_external_method_without_figures() -> None:
    with pytest.raises(ValueError, match="figure_external_method"):
        _call(
            _base_config(),
            _base_allocate_args(),
            figure_external_method={"l1_methods": ["AR(E^{CBA_FD})"]},
        )
