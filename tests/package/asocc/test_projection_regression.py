from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.regression import (
    projection_clipping_log as clip_log_mod,
)
from pyaesa.asocc.orchestration.projection.regression import (
    share_logit_time_fit_types as share_fit_types_mod,
)
from pyaesa.asocc.runtime.paths.deterministic import (
    allocate_regression_logs_dir,
)


def _state(*, runtime_proj_base: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_proj_base=runtime_proj_base,
        runtime_output_source="oecd_v2025",
    )


def _clip_counts(*, proj_base: Path) -> dict[tuple[str, ...], int]:
    return clip_log_mod.clip_counts_by_key(
        proj_base=proj_base,
        fit_start_year=2018,
        fit_end_year=2021,
        source="oecd_v2025",
        agg_version=None,
    )


def test_projection_clipping_counts_cover_empty_and_aggregated_files(tmp_path: Path) -> None:
    state = _state(runtime_proj_base=tmp_path)
    assert _clip_counts(proj_base=tmp_path) == {}

    clip_dir = allocate_regression_logs_dir(
        proj_base=tmp_path,
        source="oecd_v2025",
        agg_version=None,
    )
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / "projection_clipping_log.csv"

    clip_path.write_text("", encoding="utf-8")
    assert _clip_counts(proj_base=tmp_path) == {}

    pd.DataFrame(columns=clip_log_mod.CLIP_KEY_COLUMNS).to_csv(clip_path, index=False)
    assert _clip_counts(proj_base=tmp_path) == {}

    pd.DataFrame({"projection_branch": ["regression"], "source": ["oecd_v2025"]}).to_csv(
        clip_path,
        index=False,
    )
    assert _clip_counts(proj_base=tmp_path) == {}
    clip_path.unlink()

    clip_log_mod.write_projection_clipping_log(
        before=pd.Series([-1.0, -2.0], index=pd.Index(["FR", "FR"], name="r_p")),
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
        before=pd.Series([-3.0], index=pd.Index(["US"], name="r_p")),
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
    counts = _clip_counts(proj_base=tmp_path)
    assert counts[("regression", "oecd_v2025", "L2.a.a", "UT(FD)", "fd_rf", "FR")] == 2
    assert counts[("regression", "oecd_v2025", "L2.a.a", "UT(FD)", "fd_rf", "US")] == 1

    clip_path.unlink()
    assert _clip_counts(proj_base=tmp_path) == {}


def test_share_fit_types_cover_scalar_and_baseline_branches() -> None:
    assert share_fit_types_mod._positive_coverage_value(None) == 0  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(pd.NA) == 0  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(float("nan")) == 0  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(False) == 0  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(0) == 0  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(True) == 1  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(2) == 1  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(0.5) == 1  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(" 3.5 ") == 1  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(np.int64(1)) == 1  # noqa: SLF001
    assert share_fit_types_mod._positive_coverage_value(Decimal("NaN")) == 0  # noqa: SLF001

    baseline = share_fit_types_mod._select_baseline(  # noqa: SLF001
        modeled=["B", "A"],
        modeled_by_year={
            2018: pd.Series({"A": 0.0, "B": 1.0}),
            2019: pd.Series({"A": 0.0, "B": 1.0}),
            2020: pd.Series({"A": 0.0, "B": 1.0}),
        },
        historical_years=[2018, 2019, 2020],
        container_key=("FR",),
    )
    assert baseline == "B"

    with pytest.raises(ValueError):
        share_fit_types_mod._select_baseline(  # noqa: SLF001
            modeled=["A", "B"],
            modeled_by_year={
                2018: pd.Series({"A": 0.0, "B": 0.0}),
                2019: pd.Series({"A": 0.0, "B": 0.0}),
            },
            historical_years=[2018, 2019],
            container_key=("FR",),
        )
