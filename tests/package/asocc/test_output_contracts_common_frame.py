from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.common_frame import (
    coalesce_unique_non_null,
)
from pyaesa.asocc.runtime.output.contracts import (
    OutputRoute,
    OutputSpec,
    contract_year_columns,
    identifier_columns_from_frame,
    persisted_method_columns_for_output_spec,
)


def test_output_spec_file_name_and_format_variants() -> None:
    route = OutputRoute(
        level="L2",
        bucket="l2_vs_global",
        source="oecd_v2025",
        grouped_mode=False,
        variant_tag=None,
        ssp_scenario="SSP2",
        lcia_method=None,
        projection_subfolder=None,
    )
    spec = OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem="table_l2_asocc",
        route=route,
        scenario_dependent=True,
        identifier_columns=("r_p",),
    )

    assert spec.file_name == "table_l2_asocc__ssp2.csv"
    assert spec.file_name_for_format("csv") == "table_l2_asocc__ssp2.csv"
    assert spec.file_name_for_format("pickle") == "table_l2_asocc__ssp2.pickle"
    assert spec.file_name_for_format("parquet") == "table_l2_asocc__ssp2.parquet"


def test_output_spec_file_name_for_format_covers_suffix_variants() -> None:
    # Scenario exists but artifact is not scenario dependent.
    route_with_scenario = OutputRoute(
        level="L1",
        bucket=None,
        source="oecd_v2025",
        grouped_mode=False,
        variant_tag=None,
        ssp_scenario="SSP2",
        lcia_method=None,
        projection_subfolder=None,
    )
    spec_not_scenario_dep = OutputSpec(
        l1_l2_method="EG(Pop)",
        l2_method=None,
        l1_method="EG(Pop)",
        file_stem="table_l1_asocc",
        route=route_with_scenario,
        scenario_dependent=False,
        identifier_columns=("r_f",),
    )
    assert spec_not_scenario_dep.file_name_for_format("csv") == "table_l1_asocc.csv"

    route_with_suffix = OutputRoute(
        level="L2",
        bucket="l2_vs_global",
        source="oecd_v2025",
        grouped_mode=False,
        variant_tag=None,
        ssp_scenario="SSP2",
        lcia_method=None,
        projection_subfolder=None,
    )
    suffixed_spec = OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem="table_l2_asocc",
        route=route_with_suffix,
        scenario_dependent=True,
        identifier_columns=("r_p",),
        terminal_suffix="per_rf",
    )
    assert suffixed_spec.persisted_stem == "table_l2_asocc"
    assert suffixed_spec.file_name == "table_l2_asocc__ssp2__per_rf.csv"


def test_identifier_columns_from_frame_supports_multiindex() -> None:
    idx = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "B")], names=["r_p", "s_p"])
    frame = pd.DataFrame({2020: [1.0, 2.0]}, index=idx)
    assert identifier_columns_from_frame(frame) == ("r_p", "s_p")


def test_identifier_columns_from_frame_requires_named_levels() -> None:
    idx = pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", None])
    frame = pd.DataFrame({2020: [1.0]}, index=idx)
    with pytest.raises(ValueError):
        identifier_columns_from_frame(frame)


def test_contract_year_columns_prefers_signature_years() -> None:
    context = SimpleNamespace(
        persisted_years=[1995, 1996],
        resolved_years=[2001],
    )
    assert contract_year_columns(context) == ("1995", "1996")


def test_contract_year_columns_falls_back_to_resolved_years() -> None:
    context = SimpleNamespace(
        persisted_years=[],
        resolved_years=[2010, 2011],
    )
    assert contract_year_columns(context) == ("2010", "2011")


def test_coalesce_unique_non_null_handles_empty_unique_and_conflict() -> None:
    assert (
        coalesce_unique_non_null(
            pd.Series([pd.NA, None]),
            conflict_context="x",
        )
        is pd.NA
    )

    assert (
        coalesce_unique_non_null(
            pd.Series([pd.NA, 3.0]),
            conflict_context="x",
        )
        == 3.0
    )

    with pytest.raises(ValueError):
        coalesce_unique_non_null(
            pd.Series([1.0, 2.0]),
            conflict_context="x",
        )


def test_file_token_cover_blank_and_error_paths() -> None:
    from pyaesa.asocc.runtime.output.contracts import (
        _optional_file_token,
        join_file_owned_tokens,
    )

    assert _optional_file_token(None) is None  # noqa: SLF001
    assert _optional_file_token("   ") is None  # noqa: SLF001
    assert _optional_file_token(" _token_ ") == "token"  # noqa: SLF001

    with pytest.raises(ValueError):
        join_file_owned_tokens(None, "   ")  # noqa: SLF001


def test_persisted_method_columns_cover_l1_only_scope() -> None:
    spec = OutputSpec(
        l1_l2_method="",
        l2_method=None,
        l1_method="EG(Pop)",
        file_stem="table_l1_asocc",
        route=OutputRoute(
            level="L1",
            bucket=None,
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_f",),
    )
    assert persisted_method_columns_for_output_spec(spec) == ("l1_method",)
