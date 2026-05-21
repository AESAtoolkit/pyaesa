"""LCIA support ownership for aSoCC method routing and yearly loading."""

from typing import Literal, cast

import pandas as pd

from pyaesa.shared.lcia.paths import responsibility_periods_csv_path
from pyaesa.shared.lcia.prerequisite_tables import clean_responsibility_period_frame


def read_rps_frame(path) -> pd.DataFrame:
    """Read one responsibility-period CSV and normalize its public schema."""
    if not path.exists():
        raise FileNotFoundError(f"RPS file not found: {path}")
    frame = clean_responsibility_period_frame(frame=pd.read_csv(path), path=path)
    frame.attrs["source_csv"] = str(path)
    return frame


def load_rps_frame(*, source: str, lcia_method: str) -> pd.DataFrame:
    """Load responsibility-period settings for one LCIA method."""
    return read_rps_frame(
        responsibility_periods_csv_path(
            source=source,
            lcia_method=lcia_method,
        )
    )


def load_impact_parent_mapping(*, source: str, lcia_method: str) -> pd.Series:
    """Load the impact-to-parent mapping for one LCIA method."""
    path = responsibility_periods_csv_path(
        source=source,
        lcia_method=lcia_method,
    )
    frame = read_rps_frame(path)
    if "impact_parent" not in frame.columns:
        raise ValueError(f"RPS mapping file missing required impact_parent column: {path}")
    mapping = frame[["impact", "impact_parent"]].copy()
    impact_col = cast(pd.Series, mapping["impact"])
    parent_col = cast(pd.Series, mapping["impact_parent"])
    if bool(impact_col.isna().any()) or bool(parent_col.isna().any()):
        raise ValueError(f"RPS mapping file contains null impact/impact_parent values: {path}")
    mapping["impact"] = impact_col.astype(str)
    mapping["impact_parent"] = parent_col.astype(str)
    unique = cast(pd.DataFrame, mapping.drop_duplicates())
    series = cast(pd.Series, unique.set_index("impact")["impact_parent"])
    series.attrs["source_csv"] = str(path)
    return series


def normalize_lcia_methods(
    lcia_method: str | list[str] | None,
) -> list[str] | None:
    """Normalize `lcia_method` request values into one canonical list."""
    if lcia_method is None:
        return None
    if isinstance(lcia_method, str):
        return [lcia_method]
    lcia_methods = [
        str(method_name).strip() for method_name in lcia_method if str(method_name).strip()
    ]
    if len(lcia_methods) != len(set(lcia_methods)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for method_name in lcia_methods:
            if method_name in seen and method_name not in duplicates:
                duplicates.append(method_name)
            seen.add(method_name)
        raise ValueError(f"Duplicate lcia_method values are not allowed. Duplicates: {duplicates}.")
    return lcia_methods or None


def aggregate_frame_to_parent(
    frame: pd.DataFrame,
    impact_parent_map: pd.Series,
) -> pd.DataFrame:
    """Aggregate LCIA impacts from child to parent using the RPS mapping."""
    if "impact" not in frame.index.names:
        return frame

    source_csv = impact_parent_map.attrs.get("source_csv")
    source_hint = f" CSV: {source_csv}" if source_csv else ""
    parent_map = impact_parent_map.astype(str)
    impact_values = pd.Index(frame.index.get_level_values("impact")).astype(str)
    parent_lookup = parent_map.to_dict()
    mapped = pd.Series(
        [parent_lookup.get(str(impact), pd.NA) for impact in impact_values.to_list()],
        index=impact_values,
        dtype="object",
    )
    missing_parent = mapped.isna().to_numpy(dtype=bool)
    if bool(missing_parent.any()):
        sample = impact_values[missing_parent].astype(str).unique().tolist()[:10]
        raise ValueError(
            "Missing impact_parent mapping for LCIA impacts. "
            f"Missing impacts (sample): {sample}.{source_hint}"
        )
    mapped_index = pd.Index(mapped.astype(str), name="impact")

    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex):
        level_names = [str(name) for name in out.index.names]
        arrays: list[pd.Index] = []
        for pos, name in enumerate(level_names):
            if name == "impact":
                arrays.append(mapped_index)
            else:
                arrays.append(pd.Index(out.index.get_level_values(pos), name=name))
        out.index = pd.MultiIndex.from_arrays(arrays, names=level_names)
        return cast(
            pd.DataFrame,
            out.groupby(level=level_names, sort=False).sum(min_count=1),
        )

    out.index = mapped_index
    return cast(pd.DataFrame, out.groupby(level=0, sort=False).sum(min_count=1))


def aggregate_lcia_to_parent(
    lcia_data: dict[str, pd.DataFrame],
    impact_parent_map: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Aggregate all LCIA payload tables to the parent-impact level."""
    aggregated: dict[str, pd.DataFrame] = {}
    for key, frame in lcia_data.items():
        aggregated[key] = aggregate_frame_to_parent(frame, impact_parent_map)
    return aggregated


def initialize_pr_hr_timeseries(
    *,
    source: str,
    state,
    lcia_methods: list[str] | None,
    selected_l1: list[str],
    store: Literal["grouped", "original"] = "grouped",
) -> None:
    """Initialize PR-HR LCIA cache containers and source metadata in state."""
    if not any("PR-HR" in name for name in selected_l1):
        return
    if not lcia_methods:
        return
    target = {
        "grouped": state.lcia_timeseries,
        "original": state.lcia_timeseries_original,
    }[store]
    for method_name in lcia_methods:
        target.setdefault(method_name, {"CBA_FD": {}, "PBA": {}})
        if method_name not in state.rps_by_method:
            state.rps_by_method[method_name] = load_rps_frame(
                source=source,
                lcia_method=method_name,
            )
        if method_name not in state.cf_by_method:
            state.cf_by_method[method_name] = load_impact_parent_mapping(
                source=source,
                lcia_method=method_name,
            )
