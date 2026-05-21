"""Historical output reuse for UT projection fast paths."""

from pathlib import Path
from typing import cast

import pandas as pd

from ...common_frame import coalesce_unique_non_null
from ....runtime.paths.published import reuse_output_path_for
from ....runtime.output.contracts import join_file_owned_tokens
from pyaesa.shared.tabular.wide_tables import METHOD_IDENTITY_COLUMNS


def _normalized_reuse_lcia_key(
    *,
    l2_method: str,
    lcia_key: str | None,
) -> str | None:
    """Normalize LCIA key used by reuse caches and file stems.

    UT historical reuse payloads are not LCIA specific, so they are stored
    and loaded under a single key/file stem without LCIA suffix.
    """
    if str(l2_method).startswith("UT("):
        return None
    return lcia_key


def _reuse_file_stem(
    *,
    bucket: str,
    l2_method: str,
    lcia_key: str | None,
) -> str:
    """Return the canonical deterministic published stem for one reuse source."""
    method_token = f"l2_{l2_method}" if str(bucket).strip() == "l2_in_l1" else l2_method
    return join_file_owned_tokens(method_token, lcia_key)


def _is_historical_year(*, context, year: int) -> bool:
    """Return whether a year is inside historical MRIO coverage."""
    return int(year) in {int(value) for value in context.historical_years}


def _year_columns(df: pd.DataFrame) -> list[str]:
    """Return canonical year columns present in one wide output frame."""
    return [str(col) for col in df.columns if str(col).isdigit()]


def _normalize_single_year_frame(
    *,
    frame: pd.DataFrame,
    source_year: int,
) -> pd.DataFrame:
    """Normalize a one year indexed frame for reuse caches."""
    if frame.shape[1] != 1:
        raise ValueError(
            "Reuse source frame must have exactly one year column. "
            f"Got columns={list(frame.columns)}"
        )
    out = frame.copy()
    out.columns = [int(source_year)]
    out[out.columns[0]] = pd.to_numeric(out.iloc[:, 0], errors="raise")
    return out


def _read_wide_output(*, path: Path, output_format: str) -> pd.DataFrame:
    """Read one persisted wide output table from disk."""
    if output_format == "csv":
        return pd.read_csv(path)
    if output_format == "pickle":
        return cast(pd.DataFrame, pd.read_pickle(path))
    return cast(pd.DataFrame, pd.read_parquet(path))


def _load_source_year_from_output(
    *,
    context,
    bucket: str,
    file_stem: str,
    source_year: int,
) -> pd.DataFrame:
    """Load one historical source year frame from persisted wide outputs."""
    out_path = reuse_output_path_for(context=context, bucket=bucket, file_stem=file_stem)
    if not out_path.exists():
        raise FileNotFoundError(
            f"Historical reuse source output is missing. Expected file: {out_path}"
        )
    wide_df = _read_wide_output(path=out_path, output_format=context.output_format)
    year_col = str(int(source_year))
    if year_col not in wide_df.columns:
        raise ValueError(
            "Historical reuse source year is missing in output file. "
            f"year={source_year}, file={out_path}"
        )
    year_cols = set(_year_columns(wide_df))
    id_cols = [
        str(col)
        for col in wide_df.columns
        if str(col) not in year_cols and str(col) not in METHOD_IDENTITY_COLUMNS
    ]
    source = wide_df.loc[:, [*id_cols, year_col]].copy()
    if source.duplicated(subset=id_cols, keep=False).any():
        source = source.groupby(id_cols, dropna=False, as_index=False)[year_col].agg(
            lambda values: coalesce_unique_non_null(
                values,
                conflict_context="the same reuse source key",
            )
        )
    source[year_col] = pd.to_numeric(source[year_col], errors="raise")
    source = source.set_index(id_cols)
    source.columns = [int(source_year)]
    return source


def cache_historical_preweight(
    *,
    context,
    state,
    year: int,
    l2_method: str,
    lcia_key: str | None,
    frame: pd.DataFrame,
) -> None:
    """Cache one historical UT preweight frame for future reuse."""
    if not _is_historical_year(context=context, year=year):
        return
    cache_key = (
        "preweight",
        l2_method,
        _normalized_reuse_lcia_key(l2_method=l2_method, lcia_key=lcia_key),
        int(year),
    )
    state.ut_reuse_preweight_cache[cache_key] = _normalize_single_year_frame(
        frame=frame,
        source_year=int(year),
    )


def cache_historical_one_step_result(
    *,
    context,
    state,
    year: int,
    l2_method: str,
    lcia_key: str | None,
    frame: pd.DataFrame,
) -> None:
    """Cache one historical UT one step output frame for future reuse."""
    if not _is_historical_year(context=context, year=year):
        return
    cache_key = (
        "one_step",
        l2_method,
        _normalized_reuse_lcia_key(l2_method=l2_method, lcia_key=lcia_key),
        int(year),
    )
    state.ut_reuse_one_step_cache[cache_key] = _normalize_single_year_frame(
        frame=frame,
        source_year=int(year),
    )


def load_reuse_preweight(
    *,
    context,
    state,
    l2_method: str,
    lcia_key: str | None,
    l2_reuse_year: int,
) -> pd.DataFrame:
    """Load reusable UT preweight frame for one historical L2 reuse year."""
    normalized_lcia_key = _normalized_reuse_lcia_key(
        l2_method=l2_method,
        lcia_key=lcia_key,
    )
    cache_key = ("preweight", l2_method, normalized_lcia_key, int(l2_reuse_year))
    cached = state.ut_reuse_preweight_cache.get(cache_key)
    if cached is not None:
        return cached
    file_stem = _reuse_file_stem(
        bucket="l2_in_l1",
        l2_method=l2_method,
        lcia_key=normalized_lcia_key,
    )
    loaded = _load_source_year_from_output(
        context=context,
        bucket="l2_in_l1",
        file_stem=file_stem,
        source_year=int(l2_reuse_year),
    )
    state.ut_reuse_preweight_cache[cache_key] = loaded
    return loaded


def load_reuse_one_step_result(
    *,
    context,
    state,
    l2_method: str,
    lcia_key: str | None,
    l2_reuse_year: int,
    target_year: int,
) -> pd.DataFrame:
    """Load reusable UT one step output and remap to target year column."""
    normalized_lcia_key = _normalized_reuse_lcia_key(
        l2_method=l2_method,
        lcia_key=lcia_key,
    )
    cache_key = ("one_step", l2_method, normalized_lcia_key, int(l2_reuse_year))
    source = state.ut_reuse_one_step_cache.get(cache_key)
    if source is None:
        file_stem = _reuse_file_stem(
            bucket="l2_vs_global",
            l2_method=l2_method,
            lcia_key=normalized_lcia_key,
        )
        source = _load_source_year_from_output(
            context=context,
            bucket="l2_vs_global",
            file_stem=file_stem,
            source_year=int(l2_reuse_year),
        )
        state.ut_reuse_one_step_cache[cache_key] = source
    out = source.copy()
    out.columns = [int(target_year)]
    return out
