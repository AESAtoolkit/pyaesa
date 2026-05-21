from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.regression import (
    projection_clipping_log as clip_log_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_fit_window_log as fit_window_log_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_projection as share_mod,
)
from pyaesa.asocc.runtime.paths.deterministic import (
    projection_clipping_log_path,
    share_fit_window_log_path,
)


def _state(*, runtime_proj_base: Path) -> SimpleNamespace:
    return SimpleNamespace(
        notices_emitted=set(),
        regression_fit_cache={},
        regression_stats_rows=[],
        regression_fit_inputs_rows=[],
        regression_uncertainty_rows=[],
        mrio_units={},
        mrio_default_monetary_unit="USD_2021",
        runtime_output_source="oecd_v2025",
        runtime_proj_base=runtime_proj_base,
    )


def test_projection_and_share_fit_window_log(tmp_path: Path) -> None:
    state = _state(runtime_proj_base=tmp_path)
    clip_log_mod.write_projection_clipping_log(
        before=pd.Series([-1.0, 2.0], index=pd.Index(["FR", "US"], name="r_p")),
        source="oecd_v2025",
        projection_branch="regression",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        year=2030,
        unit="USD_2021",
        fit_start_year=2018,
        fit_end_year=2021,
        state=state,
    )
    clip_log_mod.write_projection_clipping_log(
        before=pd.Series([-3.0], index=pd.Index(["DE"], name="r_p")),
        source="oecd_v2025",
        projection_branch="regression",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_rf",
        year=2031,
        unit="USD_2021",
        fit_start_year=2018,
        fit_end_year=2021,
        state=state,
    )
    fit_window_log_mod.write_share_fit_window_log_row(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_share_sp",
        container_label="FR",
        category="A",
        baseline="B",
        fit_start_year=2018,
        fit_end_year=2021,
        years_used=[2019, 2020, 2021],
        dropped_numerator_zero_years=[2018],
        dropped_baseline_zero_years=[],
        case="subset_fit_window",
        state=state,
    )

    clip_path = projection_clipping_log_path(
        state=state,
    )
    fit_path = share_fit_window_log_path(
        state=state,
    )
    assert pd.read_csv(clip_path).shape[0] == 2
    assert pd.read_csv(fit_path).shape[0] == 1


def test_share_projection_strict_failures(tmp_path: Path) -> None:
    base_args: dict[str, Any] = dict(
        source="oecd_v2025",
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        target_object="fd_share_sp",
        target_year=2030,
        future_years=[2030],
        container_levels=[],
        category_level="s_p",
        selected_categories=None,
        selected_containers=None,
    )
    with pytest.raises(ValueError, match="positivity_counts"):
        share_mod.project_share_from_time_logit(
            **base_args,
            historical_years=[2019, 2020],
            share_by_year={
                2019: pd.Series([0.5, 0.5], index=pd.Index(["A", "B"], name="s_p")),
                2020: pd.Series([0.4, 0.6], index=pd.Index(["A", "B"], name="s_p")),
            },
            state=_state(runtime_proj_base=tmp_path),
        )
    with pytest.raises(ValueError, match="nonzero_years"):
        share_mod.project_share_from_time_logit(
            **base_args,
            historical_years=[2018, 2019, 2020, 2021],
            share_by_year={
                2018: pd.Series([0.0, 1.0], index=pd.Index(["A", "B"], name="s_p")),
                2019: pd.Series([0.3, 0.7], index=pd.Index(["A", "B"], name="s_p")),
                2020: pd.Series([0.0, 1.0], index=pd.Index(["A", "B"], name="s_p")),
                2021: pd.Series([0.2, 0.8], index=pd.Index(["A", "B"], name="s_p")),
            },
            state=_state(runtime_proj_base=tmp_path),
        )
    with pytest.raises(ValueError):
        share_mod.project_share_from_time_logit(
            **base_args,
            historical_years=[2018, 2019, 2020, 2021],
            share_by_year={
                2018: pd.Series([0.4, 0.6], index=pd.Index(["B", "C"], name="s_p")),
                2019: pd.Series([0.4, 0.0], index=pd.Index(["B", "C"], name="s_p")),
                2020: pd.Series([0.0, 0.6], index=pd.Index(["B", "C"], name="s_p")),
                2021: pd.Series([0.4, 0.6], index=pd.Index(["B", "C"], name="s_p")),
            },
            state=_state(runtime_proj_base=tmp_path),
        )
