"""Append only logging for share regression fit window diagnostics."""

import pandas as pd

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from ....runtime.paths.deterministic import share_fit_window_log_path


def write_share_fit_window_log_row(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    target_object: str,
    container_label: str,
    category: object,
    baseline: object,
    fit_start_year: int,
    fit_end_year: int,
    years_used: list[int],
    dropped_numerator_zero_years: list[int],
    dropped_baseline_zero_years: list[int],
    case: str,
    state,
) -> None:
    """Append one fit window diagnostics row for share regression handling."""
    path = share_fit_window_log_path(
        state=state,
    )
    path = ensure_file_parent(path)
    row = pd.DataFrame(
        [
            {
                "projection_branch": "regression",
                "source": str(source),
                "fu_code": str(fu_code),
                "l2_method": str(l2_method),
                "target_object": str(target_object),
                "container_label": str(container_label),
                "category": str(category),
                "baseline": str(baseline),
                "fit_start_year": int(fit_start_year),
                "fit_end_year": int(fit_end_year),
                "years_used": ",".join(str(int(year)) for year in years_used),
                "dropped_numerator_zero_years": ",".join(
                    str(int(year)) for year in dropped_numerator_zero_years
                ),
                "dropped_baseline_zero_years": ",".join(
                    str(int(year)) for year in dropped_baseline_zero_years
                ),
                "case": str(case),
            }
        ]
    )
    mode = "a" if path.exists() else "w"
    row.to_csv(path, mode=mode, index=False, header=mode == "w")
