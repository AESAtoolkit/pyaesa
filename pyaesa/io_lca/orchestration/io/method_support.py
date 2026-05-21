"""Canonical per-method aggregation and consistency ownership for IO-LCA."""

from typing import cast

import pandas as pd
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.data.loaders import YearMethodMainPayload
from pyaesa.io_lca.data.paths import IOLCAPaths, stage_results_path
from pyaesa.shared.selectors.aggregate_labels import aggregate_selector_label


def _sum_frame_columns(
    *,
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    """Group and sum selected value columns while preserving a DataFrame contract."""
    grouped = frame.groupby(group_cols, dropna=False, as_index=False)[value_cols].sum(min_count=1)
    return cast(pd.DataFrame, grouped)


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric Series with NA coerced to ``0.0``."""
    numeric = cast(pd.Series, pd.to_numeric(values, errors="raise"))
    return cast(pd.Series, numeric.fillna(0.0))


def _valid_origin_years(*, origin_frame: pd.DataFrame, lcia_method: str) -> list[int]:
    """Return validated origin years for upstream consistency checks."""
    del lcia_method
    numeric_years = cast(
        pd.Series,
        pd.to_numeric(pd.Series(origin_frame["year"], copy=False), errors="raise"),
    )
    return sorted({int(year) for year in numeric_years.dropna().tolist()})


def main_key_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return key columns for deterministic main results merges."""
    return ["lcia_method", "year", "impact", *selector_axes]


def main_result_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return the canonical persisted main-results column order."""
    return ["lcia_method", "year", "impact", *selector_axes, "lca_value", "impact_unit"]


def origin_id_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return identifier columns for wide format origin outputs."""
    return [
        "impact",
        "origin_r_p",
        "origin_s_p",
        "impact_unit",
        *selector_axes,
    ]


def origin_long_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return the canonical long-form origin column order before wide pivot."""
    return ["year", *origin_id_columns(selector_axes), "lca_value"]


def origin_ratio_group_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return grouping columns used to normalize origin shares."""
    return ["impact", "impact_unit", *selector_axes]


def stage_key_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return key columns for deterministic stage result merges."""
    return [
        *selector_axes,
        "stage",
        "stage_r_p",
        "stage_s_p",
        "linked_from_stage",
        "linked_from_r_p",
        "linked_from_s_p",
        "impact",
        "impact_unit",
    ]


def stage_public_columns(selector_axes: tuple[str, ...]) -> list[str]:
    """Return the canonical persisted stage-result column order."""
    return [
        *selector_axes,
        "stage",
        "stage_r_p",
        "stage_s_p",
        "linked_from_stage",
        "linked_from_r_p",
        "linked_from_s_p",
        "impact",
        "impact_unit",
        "direct_at_stage",
        "embedded_from_deeper_stages",
        "stage_total",
    ]


def _collapse_selector_axes(
    *,
    frame: pd.DataFrame,
    selector_axes: tuple[str, ...],
) -> pd.DataFrame:
    """Collapse selector axis values to full aggregate labels."""
    out = frame.copy()
    for axis in selector_axes:
        out[axis] = aggregate_selector_label(out[axis].tolist())
    return out


def aggregate_main(
    frame: pd.DataFrame,
    *,
    selector_axes: tuple[str, ...] = tuple(),
) -> pd.DataFrame:
    """Aggregate one main results frame across selector axes."""
    work = _collapse_selector_axes(frame=frame, selector_axes=selector_axes)
    aggregated = _sum_frame_columns(
        frame=work,
        group_cols=["lcia_method", "year", "impact", *selector_axes, "impact_unit"],
        value_cols=["lca_value"],
    )
    return aggregated.loc[:, main_result_columns(selector_axes)]


def aggregate_origin(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one origin frame across selector axes."""
    group_cols = origin_long_columns(tuple())[:-1]
    aggregated = _sum_frame_columns(frame=frame, group_cols=group_cols, value_cols=["lca_value"])
    return aggregated.loc[:, origin_long_columns(tuple())]


def aggregate_stage(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one stage frame across selector axes."""
    group_cols = stage_public_columns(tuple())[:-3]
    aggregated = _sum_frame_columns(
        frame=frame,
        group_cols=group_cols,
        value_cols=["direct_at_stage", "embedded_from_deeper_stages", "stage_total"],
    )
    return aggregated.loc[:, stage_public_columns(tuple())]


def validate_upstream_origin_matches_main(
    *,
    main_frame: pd.DataFrame,
    origin_frame: pd.DataFrame,
    selector_axes: tuple[str, ...],
    lcia_method: str,
) -> None:
    """Fail fast when origin totals do not match main IO-LCA totals."""
    if origin_frame.empty:
        return
    key_cols = ["year", *selector_axes, "impact", "impact_unit"]
    missing_main = [col for col in key_cols + ["lca_value"] if col not in main_frame.columns]
    if missing_main:
        raise ValueError(
            "Cannot validate upstream-origin consistency: "
            f"main results are missing columns {missing_main} for method '{lcia_method}'."
        )
    origin_years = _valid_origin_years(origin_frame=origin_frame, lcia_method=lcia_method)
    main_scoped = cast(
        pd.DataFrame,
        main_frame.loc[main_frame["year"].astype(int).isin(origin_years)].copy(),
    )
    main_totals = _sum_frame_columns(
        frame=main_scoped,
        group_cols=key_cols,
        value_cols=["lca_value"],
    ).rename(columns={"lca_value": "main_value"})
    origin_totals = _sum_frame_columns(
        frame=origin_frame,
        group_cols=key_cols,
        value_cols=["lca_value"],
    ).rename(columns={"lca_value": "origin_value"})
    merged = main_totals.merge(origin_totals, on=key_cols, how="outer")
    merged["main_value"] = _numeric_series(cast(pd.Series, merged["main_value"]))
    merged["origin_value"] = _numeric_series(cast(pd.Series, merged["origin_value"]))
    merged["abs_diff"] = (merged["origin_value"] - merged["main_value"]).abs()
    scale = cast(pd.Series, merged[["main_value", "origin_value"]].abs().max(axis=1)).clip(
        lower=1.0
    )
    tolerance = 1e-8 + (1e-7 * scale)
    failed = merged.loc[merged["abs_diff"] > tolerance].copy()
    if failed.empty:
        return
    failed = failed.sort_values("abs_diff", ascending=False).head(10).reset_index(drop=True)
    sample_cols = [*key_cols, "main_value", "origin_value", "abs_diff"]
    sample = {
        "columns": sample_cols,
        "values": [
            tuple(values)
            for values in failed.loc[:, sample_cols].itertuples(index=False, name=None)
        ],
    }
    max_diff = float(failed["abs_diff"].max())
    raise ValueError(
        "Upstream origin consistency check failed for method "
        f"'{lcia_method}'. Origin totals must equal main IO-LCA totals for each key. "
        f"Checked years={origin_years}. max_abs_diff={max_diff:.6e}. Sample={sample}"
    )


def to_origin_ratio_wide(
    *,
    frame: pd.DataFrame,
    selector_axes: tuple[str, ...],
) -> pd.DataFrame:
    """Convert wide origin totals into wide origin share rows.

    For each year/impact group, shares are computed as ``origin_value / total``.
    Rows where group totals are numerically zero are set to ``0.0``.
    """
    id_cols = origin_id_columns(selector_axes)
    required = set(id_cols)
    missing = [col for col in id_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Cannot compute origin ratios: missing columns {missing}.")
    year_cols = [col for col in frame.columns if col not in required]
    if not year_cols:
        return frame.copy()
    ratio = frame.copy()
    share_group_cols = origin_ratio_group_columns(selector_axes)
    for year_col in year_cols:
        values = _numeric_series(cast(pd.Series, ratio[year_col]))
        totals = cast(
            pd.Series,
            ratio.assign(_value_for_ratio=values)
            .groupby(share_group_cols, dropna=False)["_value_for_ratio"]
            .transform("sum"),
        )
        safe = cast(pd.Series, totals.abs() > 1e-12)
        ratio_col = pd.Series(0.0, index=ratio.index, dtype=float)
        ratio_col.loc[safe] = values.loc[safe] / totals.loc[safe]
        ratio[year_col] = ratio_col
    return ratio


def selector_combos(
    *,
    payload: YearMethodMainPayload,
    spec: IOLCAFUSpec,
    lcia_method: str,
    filters: dict[str, list[str] | None],
) -> pd.DataFrame:
    """Return unique selector combinations for upstream decomposition."""
    from pyaesa.io_lca.compute.main_results import build_main_results_rows

    selector_cols = list(spec.selector_axes)
    if not selector_cols:
        return pd.DataFrame([{}])
    rows = (
        build_main_results_rows(
            payload=payload,
            spec=spec,
            filters=filters,
        )
        .loc[:, selector_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if rows.empty:
        return pd.DataFrame(columns=selector_cols)
    return rows


def pending_stage_years(
    *,
    years: list[int],
    existing_years: list[int],
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
) -> list[int]:
    """Return upstream stage years that still require a write."""
    pending: list[int] = []
    for year in sorted(set(years) - set(existing_years)):
        stage_path = stage_results_path(
            paths=paths,
            source=source,
            lcia_method=lcia_method,
            year=year,
            extension=extension,
        )
        if not stage_path.exists():
            pending.append(int(year))
    return pending
