from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.regression import (
    level_ols_gdp_scaled_projection as level_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    projection_clipping_log as clip_log_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    regression_core_utils as core_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_fit_containers as share_fit_containers_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_fit_diagnostics as diagnostics_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_fit_builder as share_fit_builder_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_fit_types as share_fit_types_mod,
)


def _state(
    *,
    runtime_proj_base: Path | None = None,
    runtime_source_prefix: str | None = None,
    runtime_progress: object | None = None,
) -> SimpleNamespace:
    payload = {
        "notices_emitted": set(),
        "regression_fit_cache": {},
        "regression_stats_rows": [],
        "regression_fit_inputs_rows": [],
        "regression_uncertainty_rows": [],
        "mrio_units": {},
        "mrio_default_monetary_unit": "USD_2021",
        "runtime_output_source": "oecd_v2025",
    }
    if runtime_proj_base is not None:
        payload["runtime_proj_base"] = runtime_proj_base
    if runtime_source_prefix is not None:
        payload["runtime_source_prefix"] = runtime_source_prefix
    if runtime_progress is not None:
        payload["runtime_progress"] = runtime_progress
    return SimpleNamespace(**payload)


def test_regression_core_notice_and_unit_branches() -> None:
    persistence: list[bool] = []

    def _record_message(line: str, *, persistent: bool = True) -> None:
        assert line
        persistence.append(persistent)

    state = _state(
        runtime_source_prefix="[oecd] [pre]",
        runtime_progress=SimpleNamespace(log_message=_record_message),
    )
    core_mod.emit_regression_start_notice(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        model_type="ols_level",
        target_object="fd_rf",
        state=state,
    )
    assert persistence == [False]
    assert core_mod.future_year_range_label(future_years=[]) == ""
    assert core_mod.future_year_range_label(future_years=[2035, 2030]) == "2030-2035"
    explicit_unit_state = _state()
    explicit_unit_state.mrio_units["fd_rf"] = "M EUR"
    assert (
        core_mod.mrio_level_unit_for_target(
            target_object="fd_rf",
            state=explicit_unit_state,
        )
        == "M EUR"
    )
    with pytest.raises(ValueError):
        core_mod.mrio_level_unit_for_target(target_object="unknown_metric", state=state)


def test_share_fit_container_contracts_cover_remaining_paths() -> None:
    plain = pd.Series([1.0, 2.0], index=pd.Index(["A", "B"], name="s_p"))
    sliced_plain = share_fit_containers_mod.slice_container(
        series=plain,
        container_levels=["r_f"],
        category_level="s_p",
        container_key=("FR",),
    )
    assert float(sliced_plain.sum()) == 3.0

    multi = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [("FR", "A"), ("FR", "A"), ("US", "B")],
            names=["r_f", "s_p"],
        ),
    )
    sliced_multi = share_fit_containers_mod.slice_container(
        series=multi,
        container_levels=["r_f"],
        category_level="s_p",
        container_key=("FR",),
    )
    assert float(sliced_multi.loc["A"]) == 3.0

    cat_map_multi = share_fit_containers_mod.container_category_map(
        template=multi,
        container_levels=["r_f"],
        category_level="s_p",
    )
    assert cat_map_multi == {("FR",): ["A"], ("US",): ["B"]}
    plain_dup = pd.Series([1.0, 2.0], index=pd.Index(["A", "A"], name="s_p"))
    assert share_fit_containers_mod.container_category_map(
        template=plain_dup,
        container_levels=["r_f"],
        category_level="s_p",
    ) == {tuple(): ["A"]}
    assert share_fit_containers_mod.container_label(("FR", "A")) == "FR|A"
    assert (
        share_fit_containers_mod._container_matches_filters(
            container_key=("FR",),
            container_levels=["r_f"],
            selected_by_level={"r_f": None},
        )
        is True
    )
    assert (
        share_fit_containers_mod._container_matches_filters(
            container_key=("FR",),
            container_levels=["r_f"],
            selected_by_level={"r_f": {"US"}},
        )
        is False
    )
    filtered = share_fit_containers_mod.filter_container_map(
        container_map=cat_map_multi,
        container_levels=["r_f"],
        selected_containers={"r_f": ["US"]},
    )
    assert filtered == {("US",): ["B"]}
    assert (
        share_fit_containers_mod.filter_container_map(
            container_map=cat_map_multi,
            container_levels=["r_f"],
            selected_containers={"r_f": None},
        )
        == cat_map_multi
    )
    assert share_fit_containers_mod.container_signature(
        container_levels=["r_f", "s_p"],
        selected_containers={"r_f": ["FR"], "s_p": None},
    ) == (("r_f", ("FR",)), ("s_p", ("__ALL__",)))
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                "FR": {
                    "emit": [],
                    "baseline": None,
                    "coefs": {},
                    "structural_zero_categories": [],
                    "last_vector": pd.Series(dtype=float),
                    "all_fitted": True,
                }
            }
        )
        is None
    )
    assert share_fit_containers_mod.share_fit_map_or_none({("FR",): "bad"}) is None
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                ("FR",): {
                    "emit": [],
                    "baseline": None,
                    "coefs": [],
                    "structural_zero_categories": [],
                    "last_vector": pd.Series(dtype=float),
                    "all_fitted": True,
                }
            }
        )
        is None
    )
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                ("FR",): {
                    "emit": "bad",
                    "baseline": None,
                    "coefs": {},
                    "structural_zero_categories": [],
                    "last_vector": pd.Series(dtype=float),
                    "all_fitted": True,
                }
            }
        )
        is None
    )
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                ("FR",): {
                    "emit": [],
                    "baseline": None,
                    "coefs": {},
                    "structural_zero_categories": {},
                    "last_vector": pd.Series(dtype=float),
                    "all_fitted": True,
                }
            }
        )
        is None
    )
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                ("FR",): {
                    "emit": [],
                    "baseline": None,
                    "coefs": {},
                    "structural_zero_categories": [],
                    "last_vector": {"A": 1.0},
                    "all_fitted": True,
                }
            }
        )
        is None
    )
    assert (
        share_fit_containers_mod.share_fit_map_or_none(
            {
                ("FR",): {
                    "emit": [],
                    "baseline": None,
                    "coefs": {},
                    "structural_zero_categories": [],
                    "last_vector": pd.Series(dtype=float),
                    "all_fitted": "yes",
                }
            }
        )
        is None
    )


def test_level_projection_cache_filter_and_strict_domain_branches(
    tmp_path: Path,
) -> None:
    state = _state(runtime_proj_base=tmp_path)
    historical_years = [2017, 2018, 2019, 2020]
    history = {
        2017: pd.Series([1.0, 9.0], index=pd.Index(["FR", "US"], name="r_p")),
        2018: pd.Series([2.0, 9.0], index=pd.Index(["FR", "US"], name="r_p")),
        2019: pd.Series([np.nan, 9.0], index=pd.Index(["FR", "US"], name="r_p")),
        2020: pd.Series([4.0, 9.0], index=pd.Index(["FR", "US"], name="r_p")),
    }
    gdp = {
        2017: pd.Series([10.0, 10.0], index=pd.Index(["FR", "US"], name="r_p")),
        2018: pd.Series([11.0, 11.0], index=pd.Index(["FR", "US"], name="r_p")),
        2019: pd.Series([12.0, 12.0], index=pd.Index(["FR", "US"], name="r_p")),
        2020: pd.Series([13.0, 13.0], index=pd.Index(["FR", "US"], name="r_p")),
    }
    fit_map = level_mod._fit_ols_map_for_domains(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        historical_years=historical_years,
        history_by_year=history,
        predictor_by_year=gdp,
        selected_domains=["FR"],
        state=state,
    )
    assert set(fit_map) == {"FR"}
    fit_map_cached = level_mod._fit_ols_map_for_domains(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        historical_years=historical_years,
        history_by_year=history,
        predictor_by_year=gdp,
        selected_domains=["FR"],
        state=state,
    )
    assert fit_map_cached is fit_map

    state.regression_fit_cache.clear()
    fit_map_disk = level_mod._fit_ols_map_for_domains(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        historical_years=historical_years,
        history_by_year=history,
        predictor_by_year=gdp,
        selected_domains=["FR"],
        state=state,
    )
    assert fit_map_disk == fit_map

    projected = level_mod.project_series_from_gdp(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        target_year=2030,
        historical_years=historical_years,
        history_by_year=history,
        gdp_by_year=gdp,
        gdp_target=pd.Series(
            [np.nan, 2.0],
            index=pd.Index(["FR", "ZZ"], name="r_p"),
        ),
        selected_domains=["FR"],
        state=state,
    )
    assert np.isnan(float(projected.loc["FR"]))
    assert "ZZ" not in projected.index
    assert projected.index.name == "r_p"


def test_projection_clipping_log_early_return_on_no_negatives(tmp_path: Path) -> None:
    clip_log_mod.write_projection_clipping_log(
        before=pd.Series([1.0], index=pd.Index(["FR"], name="r_p")),
        source="oecd_v2025",
        projection_branch="regression",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        year=2030,
        unit="USD_2021",
        fit_start_year=2018,
        fit_end_year=2021,
        state=_state(runtime_proj_base=tmp_path),
    )
    path = tmp_path / "logs" / "compute_asocc" / "regression_proj" / "projection_clipping_log.csv"
    assert not path.exists()


def test_share_fit_builder_covers_selected_and_partial_paths(tmp_path: Path) -> None:
    historical_years = [2018, 2019, 2020, 2021]
    share_by_year = {
        2018: pd.Series([0.0, 0.0], index=pd.Index(["A", "B"], name="s_p")),
        2019: pd.Series([0.2, 0.8], index=pd.Index(["A", "B"], name="s_p")),
        2020: pd.Series([0.3, 0.7], index=pd.Index(["A", "B"], name="s_p")),
        2021: pd.Series([0.4, 0.6], index=pd.Index(["A", "B"], name="s_p")),
    }
    state_empty = _state(runtime_proj_base=tmp_path)
    empty = share_fit_builder_mod.build_share_fit_map_impl(
        config=share_fit_types_mod.ShareFitBuildConfig(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_share_sp",
            historical_years=historical_years,
            share_by_year=share_by_year,
            future_years=[2030],
            containers=[],
            category_level="s_p",
            selected_categories=["Z"],
            selected_containers=None,
        ),
        state=state_empty,
    )
    assert empty[tuple()]["emit"] == []

    state_partial = _state(runtime_proj_base=tmp_path)
    partial = share_fit_builder_mod.build_share_fit_map_impl(
        config=share_fit_types_mod.ShareFitBuildConfig(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_share_sp",
            historical_years=historical_years,
            share_by_year=share_by_year,
            future_years=[2030],
            containers=[],
            category_level="s_p",
            selected_categories=["A"],
            selected_containers=None,
        ),
        state=state_partial,
    )
    assert partial[tuple()]["coefs"]

    full_series = {
        2018: pd.Series([0.4, 0.6], index=pd.Index(["A", "B"], name="s_p")),
        2019: pd.Series([0.5, 0.5], index=pd.Index(["A", "B"], name="s_p")),
        2020: pd.Series([0.6, 0.4], index=pd.Index(["A", "B"], name="s_p")),
        2021: pd.Series([0.7, 0.3], index=pd.Index(["A", "B"], name="s_p")),
    }
    state_full = _state(runtime_proj_base=tmp_path)
    full = share_fit_builder_mod.build_share_fit_map_impl(
        config=share_fit_types_mod.ShareFitBuildConfig(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_share_sp",
            historical_years=historical_years,
            share_by_year=full_series,
            future_years=[2030],
            containers=[],
            category_level="s_p",
            selected_categories=None,
            selected_containers=None,
        ),
        state=state_full,
    )
    assert full[tuple()]["coefs"]


def test_share_fit_diagnostics_reuses_current_owner_contracts(tmp_path: Path) -> None:
    state = _state(runtime_proj_base=tmp_path)
    context = diagnostics_mod.ShareFitDiagnosticsContext(
        config=share_fit_types_mod.ShareFitBuildConfig(
            source="oecd_v2025",
            fu_code="L2.a.a",
            l2_method="UT(FD)",
            target_object="fd_share_sp",
            historical_years=[2018, 2019, 2020, 2021],
            share_by_year={},
            future_years=[2030],
            containers=[],
            category_level="s_p",
            selected_categories=None,
            selected_containers=None,
        ),
        fit_start=2018,
        fit_end=2021,
        state=state,
    )
    diagnostics_mod.persist_share_fit_diagnostics(
        context=context,
        payload=diagnostics_mod.ShareFitDiagnosticsPayload(
            domain_key="global",
            container_name="global",
            category="A",
            baseline="B",
            n_obs=4,
            intercept=0.2,
            slope=0.05,
            r_squared=0.8,
            p_value=0.01,
            year_center=2019.5,
            x_centered=np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float),
            y_values=np.asarray([-0.4, -0.1, 0.1, 0.3], dtype=float),
            fit_points=[
                (2018, -1.5, -0.4, 0.4, 0.4, 0.6),
                (2021, 1.5, 0.3, 0.7, 0.7, 0.3),
            ],
            valid_years=[2018, 2019, 2020, 2021],
        ),
    )
    assert len(state.regression_stats_rows) == 1
    assert len(state.regression_uncertainty_rows) == 1
    assert len(state.regression_fit_inputs_rows) == 2


def test_last_modeled_vector_covers_fallback() -> None:
    modeled_by_year = {
        2018: pd.Series([0.2, 0.8], index=pd.Index(["A", "B"], name="s_p")),
        2019: pd.Series([0.0, 0.0], index=pd.Index(["A", "B"], name="s_p")),
        2020: pd.Series([0.0, 0.0], index=pd.Index(["A", "B"], name="s_p")),
    }
    resolved = diagnostics_mod.last_modeled_vector(
        historical_years=[2018, 2019, 2020],
        modeled_by_year=modeled_by_year,
    )
    assert resolved.equals(modeled_by_year[2018])
