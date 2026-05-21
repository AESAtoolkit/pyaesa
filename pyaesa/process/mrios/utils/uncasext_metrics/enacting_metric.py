"""Enacting metrics preprocessing ownership for UNCASExt outputs."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent

from .common import (
    _all_exist,
    _get_prepared_uncasext_inputs,
    _require_dataframe,
    _set_column_names,
    _set_index_names,
    _sum_columns_by_region,
    _write_pickle,
)


def _require_lcia_attr(lcia_ext: Any, attr_name: str) -> Any:
    """Return a required LCIA attribute or raise."""
    value = getattr(lcia_ext, attr_name, None)
    if value is None:
        extension_name = str(getattr(lcia_ext, "name", "<unknown>"))
        raise ValueError(
            f"LCIA extension '{extension_name}' is missing required '{attr_name}' "
            "for UNCASExt enacting metric preprocessing."
        )
    return value


def _require_lcia_attr_df(lcia_ext: Any, attr_name: str) -> pd.DataFrame:
    """Return required LCIA attribute as DataFrame."""
    return _require_dataframe(
        _require_lcia_attr(lcia_ext, attr_name), label=f"LCIA attribute '{attr_name}'"
    )


def _resolve_single_mrio_unit(iosys: Any) -> str:
    """Return one canonical MRIO monetary unit from ``iosys.unit``."""
    unit_obj = getattr(iosys, "unit", None)
    if unit_obj is None:
        raise ValueError(
            "Parsed MRIO is missing the unit table required to label UNCASExt enacting metric "
            "metric outputs."
        )
    if isinstance(unit_obj, pd.Series):
        values = unit_obj.astype(str).str.strip()
    elif isinstance(unit_obj, pd.DataFrame):
        values = unit_obj.stack().astype(str).str.strip()
    else:
        values = pd.Series([str(unit_obj).strip()])
    values = values[(values != "") & (values.str.lower() != "nan")]
    unique_units = sorted(set(values.tolist()))
    if not unique_units:
        raise ValueError("iosys.unit does not contain a usable unit label")
    if len(unique_units) != 1:
        raise ValueError(f"iosys.unit must contain one unique monetary unit, got {unique_units}")
    return unique_units[0]


def _build_enacting_metric_units_payload(
    *,
    iosys: Any,
    lcia_method_specs: list[str],
    lcia_units_by_method: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    """Build JSON serializable unit payload for enacting metrics."""
    monetary_unit = _resolve_single_mrio_unit(iosys)
    mrio_metric_units = {
        "fd_rf": monetary_unit,
        "gva_rp": monetary_unit,
        "fd_rp_sp_rf": monetary_unit,
        "fd_rp_sp": monetary_unit,
        "fd_rf_sp": monetary_unit,
        "gva_rp_sp": monetary_unit,
        "x_rp_sp": monetary_unit,
        "x_rp_sp_rc": monetary_unit,
        "x_rc_sp": monetary_unit,
    }
    lcia_payload: dict[str, dict[str, str]] = {}
    unit_maps = lcia_units_by_method or {}
    for lcia_method in lcia_method_specs:
        lcia_unit_map = unit_maps[lcia_method]
        cleaned: dict[str, str] = {}
        for impact, unit in lcia_unit_map.items():
            impact_label = str(impact).strip()
            unit_label = str(unit).strip()
            if not impact_label or not unit_label:
                continue
            cleaned[impact_label] = unit_label
        lcia_payload[lcia_method] = dict(sorted(cleaned.items()))
    return {
        "mrio_default_monetary": monetary_unit,
        "mrio_by_metric": mrio_metric_units,
        "lcia_by_method": lcia_payload,
    }


def _write_enacting_metric_units_json(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Persist enacting metric unit payload to ``path`` as JSON."""
    path = ensure_file_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _first_seen(values: pd.Index) -> list[str]:
    """Return string labels in first seen order."""
    return list(dict.fromkeys(str(value) for value in values.to_numpy()))


def _sum_product_rows_by_label(
    values: np.ndarray,
    labels: np.ndarray,
    ordered_labels: list[str],
) -> np.ndarray:
    """Return rows from ``values`` summed by product labels."""
    out = np.empty((len(ordered_labels), values.shape[1]), dtype=float)
    for idx, label in enumerate(ordered_labels):
        out[idx, :] = values[labels == label, :].sum(axis=0)
    return out


def _sum_impact_product_rows_by_label(
    factors: np.ndarray,
    activity: np.ndarray,
    labels: np.ndarray,
    ordered_labels: list[str],
) -> np.ndarray:
    """Return impact by region arrays summed by product labels."""
    out = np.empty((factors.shape[0], len(ordered_labels), activity.shape[1]), dtype=float)
    for idx, label in enumerate(ordered_labels):
        label_mask = labels == label
        out[:, idx, :] = factors[:, label_mask] @ activity[label_mask, :]
    return out


def _impact_product_region_matrix(
    *,
    factors: np.ndarray,
    activity: np.ndarray,
) -> np.ndarray:
    """Return the persisted impact product by region matrix without a tensor copy."""
    out = np.empty((factors.shape[0] * factors.shape[1], activity.shape[1]), dtype=float)
    for region_idx in range(activity.shape[1]):
        out[:, region_idx] = (factors * activity[:, region_idx][None, :]).reshape(-1)
    return out


def _precompute_enacting_metrics_uncasext(
    *,
    iosys: Any,
    saved_dir: Path,
    source_key: str,
    refresh: bool,
    lcia_methods: list[str] | None,
    matrix_version: str | None,
    lcia_units_by_method: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    """Compute/persist enacting metrics and return unit metadata payload."""
    base = saved_dir / "enacting_metrics"
    out_l1 = base / "level_1"
    out_l2 = base / "level_2"
    units_path = base / "units.json"
    out_l1 = ensure_dir(out_l1)
    out_l2 = ensure_dir(out_l2)

    is_exio = str(source_key).lower().startswith("exio")
    expected_l1 = ["fd_rf", "gva_rp"]
    expected_l2 = ["fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp", "gva_rp_sp"]
    lcia_method_specs = []
    if is_exio and lcia_methods:
        lcia_method_specs = [
            str(lcia_method).strip() for lcia_method in lcia_methods if str(lcia_method).strip()
        ]
    units_payload = _build_enacting_metric_units_payload(
        iosys=iosys,
        lcia_method_specs=lcia_method_specs,
        lcia_units_by_method=lcia_units_by_method,
    )

    if not refresh:
        l1_paths = [out_l1 / f"{name}.pickle" for name in expected_l1]
        l2_paths = [out_l2 / f"{name}.pickle" for name in expected_l2]
        if is_exio:
            l1_paths.extend(
                [
                    out_l1 / lcia_method / f"{name}.pickle"
                    for lcia_method in lcia_method_specs
                    for name in ["e_cba_fd_reg", "e_pba_reg"]
                ]
            )
            l2_paths.extend(
                [
                    out_l2 / lcia_method / f"{name}.pickle"
                    for lcia_method in lcia_method_specs
                    for name in [
                        "e_pba_rp_sp",
                        "e_cba_fd_rp_sp",
                        "e_cba_fd_rp_sp_rf",
                        "e_cba_td_rp_sp_rc",
                        "e_cba_td_rp_sp",
                        "e_cba_fd_rf_sp",
                        "e_cba_td_rc_sp",
                    ]
                ]
            )
        if _all_exist(l1_paths) and _all_exist(l2_paths):
            _write_enacting_metric_units_json(path=units_path, payload=units_payload)
            return units_payload

    prepared = _get_prepared_uncasext_inputs(
        iosys,
        source_key=source_key,
        matrix_version=matrix_version,
        saved_dir=saved_dir,
    )
    x_vec = prepared.x_vec
    y_fd = prepared.y_fd
    regions = list(y_fd.columns)
    z_reg = prepared.z_reg

    fd_rp_sp_rf = y_fd
    fd_np = fd_rp_sp_rf.to_numpy(dtype=float)
    prod_index = x_vec.index
    prod_regions = prod_index.get_level_values("r_p")
    prod_sectors = prod_index.get_level_values("s_p")
    sector_labels = _first_seen(prod_sectors)
    sector_array = np.asarray([str(value) for value in prod_sectors.to_numpy()], dtype=object)
    fd_rp_sp = pd.Series(
        fd_np.sum(axis=1),
        index=prod_index,
    )
    fd_product_total = fd_np.sum(axis=1)
    td_supply = fd_np + z_reg.to_numpy(dtype=float)
    td_product_total = td_supply.sum(axis=1)
    fd_rp_sp = _set_index_names(fd_rp_sp, ["r_p", "s_p"])
    fd_rf = pd.Series(fd_np.sum(axis=0), index=pd.Index(regions, name="r_f"))
    fd_by_sector = _sum_product_rows_by_label(fd_np, sector_array, sector_labels)
    fd_rf_sp = pd.Series(
        fd_by_sector.T.reshape(-1),
        index=pd.MultiIndex.from_product([regions, sector_labels], names=["r_f", "s_p"]),
    )
    gva_rp_sp = prepared.gva_by_prod
    region_labels = _first_seen(prod_regions)
    region_array = np.asarray([str(value) for value in prod_regions.to_numpy()], dtype=object)
    gva_np = gva_rp_sp.to_numpy(dtype=float)
    gva_rp = pd.Series(
        [float(gva_np[region_array == region].sum()) for region in region_labels],
        index=pd.Index(region_labels, name="r_p"),
    )
    _write_pickle(out_l1 / "fd_rf.pickle", fd_rf)
    _write_pickle(out_l1 / "gva_rp.pickle", gva_rp)
    _write_pickle(out_l2 / "fd_rp_sp_rf.pickle", fd_rp_sp_rf)
    _write_pickle(out_l2 / "fd_rp_sp.pickle", fd_rp_sp)
    _write_pickle(out_l2 / "fd_rf_sp.pickle", fd_rf_sp)
    _write_pickle(out_l2 / "gva_rp_sp.pickle", gva_rp_sp)
    _write_enacting_metric_units_json(path=units_path, payload=units_payload)

    if not is_exio or not lcia_method_specs:
        return units_payload

    for lcia_method in lcia_method_specs:
        lcia_ext = getattr(iosys, lcia_method, None)

        l1_method_dir = out_l1 / lcia_method
        l2_method_dir = out_l2 / lcia_method
        l1_method_dir = ensure_dir(l1_method_dir)
        l2_method_dir = ensure_dir(l2_method_dir)

        e_pba_reg = _set_index_names(_require_lcia_attr_df(lcia_ext, "D_pba_reg"), ["impact"])
        e_pba_reg = _set_column_names(e_pba_reg, ["r_p"])
        _write_pickle(l1_method_dir / "e_pba_reg.pickle", e_pba_reg)

        m_mat = _set_index_names(_require_lcia_attr_df(lcia_ext, "M"), ["impact"])
        m_mat = _set_column_names(m_mat, ["r_p", "s_p"])
        d_pba = _set_index_names(_require_lcia_attr_df(lcia_ext, "D_pba"), ["impact"])
        d_pba = _set_column_names(d_pba, ["r_p", "s_p"])
        _write_pickle(l2_method_dir / "e_pba_rp_sp.pickle", d_pba)

        m_np = m_mat.to_numpy(dtype=float)
        num_impacts, num_prod = m_np.shape

        impact = np.repeat(m_mat.index.to_numpy(), num_prod)
        prod_region = np.tile(prod_regions.to_numpy(), num_impacts)
        prod_sector = np.tile(prod_sectors.to_numpy(), num_impacts)
        idx = pd.MultiIndex.from_arrays(
            [impact, prod_region, prod_sector],
            names=["impact", "r_p", "s_p"],
        )

        e_cba_fd_rp_sp_rf = pd.DataFrame(
            _impact_product_region_matrix(factors=m_np, activity=fd_np),
            index=idx,
            columns=pd.Index(regions, name="r_f"),
        )
        _write_pickle(l2_method_dir / "e_cba_fd_rp_sp_rf.pickle", e_cba_fd_rp_sp_rf)

        e_cba_fd_rp_sp = pd.DataFrame(
            m_np * fd_product_total[None, :],
            index=m_mat.index,
            columns=prod_index,
        )
        _write_pickle(l2_method_dir / "e_cba_fd_rp_sp.pickle", e_cba_fd_rp_sp)

        fd_by_sector_by_impact = _sum_impact_product_rows_by_label(
            m_np,
            fd_np,
            sector_array,
            sector_labels,
        )
        e_cba_fd_rf_sp = pd.DataFrame(
            fd_by_sector_by_impact.transpose(0, 2, 1).reshape(num_impacts, -1),
            index=m_mat.index,
            columns=pd.MultiIndex.from_product([regions, sector_labels], names=["r_f", "s_p"]),
        )
        _write_pickle(l2_method_dir / "e_cba_fd_rf_sp.pickle", e_cba_fd_rf_sp)

        e_cba_fd_reg_calc = pd.DataFrame(
            m_np @ fd_np,
            index=m_mat.index,
            columns=pd.Index(regions, name="r_f"),
        )
        f_y = _require_lcia_attr_df(lcia_ext, "F_Y")
        fy_reg = _sum_columns_by_region(f_y).loc[:, regions]
        e_cba_fd_reg_calc = pd.DataFrame(
            e_cba_fd_reg_calc.to_numpy(dtype=float) + fy_reg.to_numpy(dtype=float),
            index=m_mat.index,
            columns=pd.Index(regions, name="r_f"),
        )
        e_cba_fd_reg_calc = _set_index_names(e_cba_fd_reg_calc, ["impact"])
        e_cba_fd_reg_calc = _set_column_names(e_cba_fd_reg_calc, ["r_f"])
        _write_pickle(l1_method_dir / "e_cba_fd_reg.pickle", e_cba_fd_reg_calc)

        e_cba_td_rp_sp_rc = pd.DataFrame(
            _impact_product_region_matrix(factors=m_np, activity=td_supply),
            index=idx,
            columns=pd.Index(regions, name="r_c"),
        )
        _write_pickle(l2_method_dir / "e_cba_td_rp_sp_rc.pickle", e_cba_td_rp_sp_rc)

        e_cba_td_rp_sp = pd.DataFrame(
            m_np * td_product_total[None, :],
            index=m_mat.index,
            columns=prod_index,
        )
        _write_pickle(l2_method_dir / "e_cba_td_rp_sp.pickle", e_cba_td_rp_sp)

        td_by_sector_by_impact = _sum_impact_product_rows_by_label(
            m_np,
            td_supply,
            sector_array,
            sector_labels,
        )
        e_cba_td_rc_sp = pd.DataFrame(
            td_by_sector_by_impact.transpose(0, 2, 1).reshape(num_impacts, -1),
            index=m_mat.index,
            columns=pd.MultiIndex.from_product([regions, sector_labels], names=["r_c", "s_p"]),
        )
        _write_pickle(l2_method_dir / "e_cba_td_rc_sp.pickle", e_cba_td_rc_sp)
    return units_payload
