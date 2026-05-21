"""Core EXIO characterization ownership.

This module contains pure characterization logic and extension table
transformations used by ``exio_parser`` wrappers.
"""

from typing import Any, Callable, Iterable, Sequence, Tuple, cast

import pandas as pd
import pymrio
from pymrio.tools.iomath import calc_M


def _requested_extensions_from_matrix(char_df: pd.DataFrame) -> list[str]:
    """Return unique extension names referenced by ``char_df``."""
    return sorted({str(value) for value in char_df["extension"].dropna()})


def _normalize_label(label: str) -> str:
    """Return a normalized label."""
    return " ".join(str(label).strip().replace("_", " ").lower().split())


def _normalized_unit_series(unit_obj: Any) -> pd.Series | None:
    """Return one normalized stressor unit Series when available."""
    if isinstance(unit_obj, pd.DataFrame) and "unit" in unit_obj.columns:
        unit_map = pd.Series(unit_obj["unit"], copy=False).astype(str).str.strip()
        unit_map = unit_map.rename_axis(index="stressor")
        unit_map.index = unit_map.index.astype(str).str.strip()
        return unit_map
    if isinstance(unit_obj, pd.Series):
        unit_map = unit_obj.astype(str).str.strip()
        unit_map.index = unit_map.index.astype(str).str.strip()
        return unit_map
    return None


def _sum_columns_by_region(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Group product or final demand columns by canonical region labels."""
    del context
    frame_t = cast(pd.DataFrame, frame.T)
    grouped = cast(pd.DataFrame, frame_t.groupby(level="region", sort=False).sum())
    return cast(pd.DataFrame, grouped.T)


def _replace_na_in_zero_output_columns(
    frame: pd.DataFrame,
    x_series: pd.Series,
) -> pd.DataFrame:
    """Return ``frame`` with missing values replaced only for zero output sectors.

    In MRIO accounting, a sector with zero total output is inactive for the year.
    Missing direct satellite entries in such columns should not propagate as
    undefined intensities through ``S = F / x`` and then contaminate the full
    multiplier row via ``M = S . L``. For inactive sectors, the operationally
    correct direct contribution is zero while missing values in active sectors
    remain an error signal and are left untouched.
    """
    x_aligned = x_series.reindex(frame.columns)
    zero_output_columns = pd.Series(x_aligned.eq(0), index=frame.columns, copy=False)
    if not zero_output_columns.any():
        return frame

    sanitized = frame.copy()
    target_columns = frame.columns[zero_output_columns.to_numpy(dtype=bool)]
    sanitized.loc[:, target_columns] = sanitized.loc[:, target_columns].fillna(0.0)
    return sanitized


def _collect_extensions(
    io_system: pymrio.IOSystem,
    requested: list[str],
) -> Tuple[list[pymrio.Extension], list[str]]:
    """Map requested labels to ``pymrio.Extension`` objects."""
    available = list(io_system.get_extensions(data=True))
    instance_names = list(io_system.get_extensions(instance_names=True))
    exact_lookup = {}
    normalized_lookup = {}
    for extension, instance_name in zip(available, instance_names):
        ext = cast(Any, extension)
        exact_lookup[ext.name] = extension
        normalized_lookup[_normalize_label(ext.name)] = extension
        normalized_lookup[_normalize_label(instance_name)] = extension

    resolved = []
    missing = []
    for name in requested:
        if name in exact_lookup:
            resolved.append(exact_lookup[name])
            continue
        normalized = _normalize_label(name)
        match = normalized_lookup.get(normalized)
        if match is None:
            missing.append(name)
        else:
            resolved.append(match)

    return resolved, missing


def _prune_extensions(io_system: pymrio.IOSystem, keep: set[str]) -> list[str]:
    """Remove extension instances not listed in ``keep``."""
    removed: list[str] = []
    for inst_name in list(io_system.get_extensions(instance_names=True)):
        if inst_name not in keep:
            delattr(io_system, inst_name)
            removed.append(inst_name)
    return removed


def _build_characterization_validation(
    *,
    char_matrix: pd.DataFrame,
    extensions: Sequence[pymrio.Extension],
) -> pd.DataFrame:
    """Build a validation DataFrame compatible with ``_ensure_validation_success``."""
    validation = pd.DataFrame(
        {
            "stressor": char_matrix["stressor"].astype(str).str.strip(),
            "extension": char_matrix["extension"].astype(str).str.strip(),
        }
    )
    validation["reason"] = ""
    validation["error_extension"] = False
    validation["error_stressor"] = False
    validation["error_unit_stressor"] = False

    ext_lookup: dict[str, pymrio.Extension] = {
        cast(str, cast(Any, ext).name): ext for ext in extensions
    }
    ext_names = sorted(ext_lookup.keys())
    validation["error_extension"] = ~validation["extension"].isin(ext_names)

    has_stressor_unit = "stressor_unit" in char_matrix.columns
    provided_units = (
        char_matrix["stressor_unit"].astype(str).str.strip()
        if has_stressor_unit
        else pd.Series("", index=char_matrix.index, dtype=str)
    )
    for ext_name, ext in ext_lookup.items():
        mask = validation["extension"] == ext_name
        if not mask.any():
            continue

        f_obj = getattr(ext, "F", None)
        if not isinstance(f_obj, pd.DataFrame):
            validation.loc[mask, "error_stressor"] = True
            continue
        stressor_index = f_obj.index.to_series().astype(str).str.strip()
        stressor_set = set(stressor_index.tolist())

        stressors = validation.loc[mask, "stressor"]
        stressor_missing = ~stressors.isin(stressor_set)
        validation.loc[mask, "error_stressor"] = stressor_missing.to_numpy()

        if has_stressor_unit:
            unit_obj = getattr(ext, "unit", None)
            unit_map = _normalized_unit_series(unit_obj)
            if unit_map is not None:
                expected_unit = stressors.map(unit_map)
                mismatch = (
                    expected_unit.notna()
                    & (provided_units.loc[mask] != "")
                    & (provided_units.loc[mask] != expected_unit)
                )
                validation.loc[mask, "error_unit_stressor"] = mismatch.to_numpy()

    reason = pd.Series("", index=validation.index, dtype=str)
    reason = reason.mask(validation["error_extension"], "missing extension")
    reason = reason.mask(
        (~validation["error_extension"]) & validation["error_stressor"],
        "missing stressor",
    )
    reason = reason.mask(validation["error_unit_stressor"], "stressor unit mismatch")
    validation["reason"] = reason
    return validation


def _build_impact_units(char_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return unit table for characterized impacts."""
    units = char_matrix.loc[:, ["impact", "impact_unit"]].dropna(subset=["impact"]).copy()
    units["impact"] = units["impact"].astype(str).str.strip()
    units["impact_unit"] = units["impact_unit"].astype(str).str.strip()
    units = units[units["impact"] != ""]

    duplicates = (
        units.groupby("impact")["impact_unit"].nunique(dropna=False).loc[lambda series: series > 1]
    )
    if not duplicates.empty:
        conflict_impacts = duplicates.index.tolist()
        source_path = char_matrix.attrs.get("source_path", "<unknown>")
        raise ValueError(
            "Inconsistent impact units in characterization matrix. "
            f"File={source_path}. Conflicting impacts={conflict_impacts[:20]}."
        )

    unit_table = units.drop_duplicates(subset=["impact"], keep="first").set_index("impact")
    unit_table.columns = pd.Index(["unit"])
    unit_table.index.name = "impact"
    return unit_table


def _project_extension_characterization(
    *,
    extension: pymrio.Extension,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Project one extension into characterized impacts via stressor factors."""
    f_obj = cast(pd.DataFrame, extension.F)

    factors_clean = factors.loc[:, ["stressor", "impact", "factor"]].copy()
    factors_clean["stressor"] = factors_clean["stressor"].astype(str).str.strip()
    factors_clean["impact"] = factors_clean["impact"].astype(str).str.strip()
    factors_clean["factor"] = pd.Series(
        pd.to_numeric(factors_clean["factor"], errors="raise"),
        copy=False,
    ).fillna(0.0)
    factors_clean = factors_clean.drop_duplicates(subset=["stressor", "impact"], keep="first")

    factor_matrix = (
        factors_clean.set_index(["stressor", "impact"])["factor"].unstack("impact").fillna(0.0)
    )
    factor_values = factor_matrix.to_numpy(dtype=float, copy=False).T
    f_work = f_obj.copy(deep=False)
    f_work.index = f_work.index.astype(str).str.strip()
    f_aligned = f_work.reindex(index=factor_matrix.index, fill_value=0.0)
    f_values = factor_values @ f_aligned.to_numpy(dtype=float, copy=False)
    f_char = pd.DataFrame(
        f_values,
        index=pd.Index(factor_matrix.columns, name="impact"),
        columns=f_aligned.columns,
    )

    fy_obj = getattr(extension, "F_Y", None)
    if not isinstance(fy_obj, pd.DataFrame):
        return f_char, None
    fy_work = fy_obj.copy(deep=False)
    fy_work.index = fy_work.index.astype(str).str.strip()
    fy_aligned = fy_work.reindex(index=factor_matrix.index, fill_value=0.0)
    fy_values = factor_values @ fy_aligned.to_numpy(dtype=float, copy=False)
    fy_char = pd.DataFrame(
        fy_values,
        index=pd.Index(factor_matrix.columns, name="impact"),
        columns=fy_aligned.columns,
    )
    return f_char, fy_char


def _direct_characterize_extensions(
    *,
    extensions: Sequence[pymrio.Extension],
    char_matrix: pd.DataFrame,
    new_extension_name: str,
) -> pymrio.Extension:
    """Create characterized extension by direct stressor factor projection."""
    ext_lookup = {cast(str, cast(Any, ext).name): ext for ext in extensions}
    f_total: pd.DataFrame | None = None
    fy_total: pd.DataFrame | None = None

    for ext_name in _requested_extensions_from_matrix(char_matrix):
        ext = ext_lookup[ext_name]
        factors = char_matrix.loc[char_matrix["extension"].astype(str).str.strip() == ext_name]
        f_part, fy_part = _project_extension_characterization(
            extension=ext,
            factors=factors,
        )
        f_total = f_part if f_total is None else f_total.add(f_part, fill_value=0.0)
        if fy_part is not None:
            fy_total = fy_part if fy_total is None else fy_total.add(fy_part, fill_value=0.0)

    unit_table = _build_impact_units(char_matrix)
    return pymrio.Extension(
        name=new_extension_name,
        F=f_total,
        F_Y=fy_total,
        unit=unit_table,
    )


def _characterize_exiobase_io_core(
    io_system: pymrio.IOSystem,
    *,
    char_matrix: pd.DataFrame,
    new_extension_name: str,
    retain_instances: Iterable[str],
    prune: bool,
    validate: Callable[[pd.DataFrame], None],
) -> tuple[list[str], list[str]]:
    """Attach characterized extension and return kept/requested extension lists."""
    requested_ext_names = _requested_extensions_from_matrix(char_matrix)
    extensions, _ = _collect_extensions(io_system, list(requested_ext_names))

    validation = _build_characterization_validation(
        char_matrix=char_matrix,
        extensions=extensions,
    )
    validate(validation)

    characterized_extension = _direct_characterize_extensions(
        extensions=extensions,
        char_matrix=char_matrix,
        new_extension_name=new_extension_name,
    )
    setattr(io_system, new_extension_name, characterized_extension)

    keep_set = set(retain_instances)
    keep_set.add(new_extension_name)
    if prune:
        removed = _prune_extensions(io_system, keep_set)
        keep_instances = sorted(
            keep_set | {inst for inst in retain_instances if inst not in removed}
        )
    else:
        keep_instances = sorted(keep_set)
    return keep_instances, requested_ext_names


def _find_missing_characterization_extensions(
    io_system: pymrio.IOSystem,
    requested_extensions: Sequence[str],
) -> list[str]:
    """Return requested extensions absent from ``io_system``."""
    if not requested_extensions:
        return []
    _, missing = _collect_extensions(io_system, list(requested_extensions))
    return missing


def _retain_extension_instances(
    io_system: pymrio.IOSystem,
    retain_instances: Iterable[str],
) -> list[str]:
    """Remove EXIO satellite accounts other than ``retain_instances``."""
    return _prune_extensions(io_system, set(retain_instances))


def _calc_characterized_extensions_minimal(
    io_system: pymrio.IOSystem,
    lcia_method_names: Sequence[str],
    *,
    keep_direct_intensities: bool,
) -> None:
    """Compute only LCIA attributes required by UNCASExt enacting metric outputs."""
    if not lcia_method_names:
        return

    x_obj = cast(pd.DataFrame, io_system.x)
    l_obj = cast(pd.DataFrame, io_system.L)
    x_series = x_obj.iloc[:, 0]

    for lcia_method in lcia_method_names:
        ext = getattr(io_system, lcia_method)
        f_obj = cast(pd.DataFrame, ext.F)

        sanitized_f = _replace_na_in_zero_output_columns(f_obj, x_series)
        s_mat = cast(pd.DataFrame, pymrio.calc_A(sanitized_f, x_series))
        ext.S = s_mat if keep_direct_intensities else None
        ext.M = cast(pd.DataFrame, calc_M(s_mat, l_obj))
        if not keep_direct_intensities:
            ext.F = None

        x_aligned = x_series.reindex(s_mat.columns)
        d_pba = s_mat.mul(x_aligned, axis=1)
        ext.D_pba = d_pba

        d_pba_reg = _sum_columns_by_region(d_pba, context="D_pba product axis")

        f_y = getattr(ext, "F_Y", None)
        if isinstance(f_y, pd.DataFrame):
            f_y_reg = _sum_columns_by_region(f_y, context="F_Y final demand axis")
            d_pba_reg = d_pba_reg.add(f_y_reg, fill_value=0.0)
        ext.D_pba_reg = d_pba_reg
