"""Persistence ownership used by MRIO processing."""

import json
import pickle
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
import pymrio
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent

from ..io.paths import (
    _get_preclip_calc_all_marker_path,
    _get_preclip_dir,
    _get_preclip_extensions_dir,
    _get_root_extensions_dir,
)


def save_minimal_core_pickles(
    *,
    iosys: pymrio.IOSystem,
    saved_dir: Path,
    core_matrices: Sequence[str],
) -> None:
    """Persist selected UNCASExt core matrices at the saved year root.

    Args:
        iosys: Computed IOSystem for one year.
        saved_dir: Target processed year directory.
        core_matrices: Core matrix names to serialize as ``<name>.pickle``.
    """
    saved_dir = ensure_dir(saved_dir)
    for name in core_matrices:
        matrix_name = str(name).strip()
        if not matrix_name:
            continue
        payload = getattr(iosys, matrix_name, None)
        if payload is None:
            raise ValueError(f"Cannot persist core matrix '{matrix_name}': missing on IOSystem.")
        out_path = saved_dir / f"{matrix_name}.pickle"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_preclip_core_pickles(
    *,
    iosys: pymrio.IOSystem,
    saved_dir: Path,
    core_matrices: Sequence[str],
) -> None:
    """Persist selected core matrices under ``preclip/``."""
    preclip_dir = _get_preclip_dir(saved_dir)
    preclip_dir = ensure_dir(preclip_dir)
    for name in core_matrices:
        matrix_name = str(name).strip()
        if not matrix_name:
            continue
        payload = getattr(iosys, matrix_name, None)
        if payload is None:
            raise ValueError(f"Cannot persist core matrix '{matrix_name}': missing on IOSystem.")
        out_path = preclip_dir / f"{matrix_name}.pickle"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pyaesa_core_outputs_exist(saved_dir: Path, *, core_matrices: Sequence[str]) -> bool:
    """Return True when all required UNCASExt core pickles are present at root."""
    if not saved_dir.exists():
        return False
    required = [str(name).strip() for name in core_matrices if str(name).strip()]
    return all((saved_dir / f"{name}.pickle").exists() for name in required)


def preclip_core_outputs_exist(saved_dir: Path, *, core_matrices: Sequence[str]) -> bool:
    """Return True when all required preclip core pickles are present."""
    preclip_dir = _get_preclip_dir(saved_dir)
    required = [str(name).strip() for name in core_matrices if str(name).strip()]
    return all((preclip_dir / f"{name}.pickle").exists() for name in required)


def _normalize_extension_payload(
    extension_payload: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    """Return normalized ``extension -> matrices`` payload for file checks."""
    if extension_payload is None:
        return {}

    normalized: dict[str, list[str]] = {}
    for raw_ext_name, raw_matrices in extension_payload.items():
        ext_name = str(raw_ext_name).strip()
        if not ext_name:
            continue
        matrix_values: list[str] = []
        candidate = raw_matrices
        if isinstance(raw_matrices, Mapping):
            candidate = raw_matrices.get("matrices")
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            seen: set[str] = set()
            for raw_matrix in candidate:
                matrix_name = str(raw_matrix).strip()
                if not matrix_name or matrix_name in seen:
                    continue
                seen.add(matrix_name)
                matrix_values.append(matrix_name)
        normalized[ext_name] = matrix_values
    return normalized


def _extension_files_exist(
    *,
    extensions_dir: Path,
    extension_payload: dict[str, list[str]],
) -> bool:
    """Return True when all expected extension pickle files are present."""
    if not extension_payload:
        return True
    if not extensions_dir.exists():
        return False
    for ext_name, matrix_names in extension_payload.items():
        ext_dir = extensions_dir / ext_name
        if not ext_dir.exists():
            return False
        if not matrix_names:
            if not any(ext_dir.glob("*.pickle")):
                return False
            continue
        for matrix_name in matrix_names:
            if not (ext_dir / f"{matrix_name}.pickle").exists():
                return False
    return True


def _marker_extension_payload(saved_dir: Path) -> dict[str, list[str]] | None:
    """Return marker derived preclip extension payload."""
    marker_path = _get_preclip_calc_all_marker_path(saved_dir)
    if not marker_path.exists():
        return None
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_payload = cast(Mapping[str, Any], marker_data)
    raw_extensions = cast(Sequence[Any], marker_payload["extensions"])
    payload: dict[str, list[str]] = {}
    for raw_ext in raw_extensions:
        ext_name = str(raw_ext).strip()
        if ext_name:
            payload[ext_name] = []
    return payload


def preclip_extension_outputs_exist(
    saved_dir: Path,
    *,
    extension_payload: Mapping[str, Any] | None,
) -> bool:
    """Return True when expected preclip extension pickles are present."""
    normalized_payload = _normalize_extension_payload(extension_payload)
    if normalized_payload:
        return _extension_files_exist(
            extensions_dir=_get_preclip_extensions_dir(saved_dir),
            extension_payload=normalized_payload,
        )
    marker_payload = _marker_extension_payload(saved_dir)
    if marker_payload is None:
        return False
    return _extension_files_exist(
        extensions_dir=_get_preclip_extensions_dir(saved_dir),
        extension_payload=marker_payload,
    )


def pyaesa_extension_outputs_exist(
    saved_dir: Path,
    *,
    extension_payload: Mapping[str, Any] | None,
) -> bool:
    """Return True when expected UNCASExt extension pickles are present."""
    normalized_payload = _normalize_extension_payload(extension_payload)
    return _extension_files_exist(
        extensions_dir=_get_root_extensions_dir(saved_dir),
        extension_payload=normalized_payload,
    )


def _extension_serializable_items(ext: Any) -> Dict[str, Any]:
    """Return extension attributes that should be serialized for preclip payloads."""
    items: Dict[str, Any] = {}
    for attr_name, value in vars(ext).items():
        name = str(attr_name).strip()
        if not name or name.startswith("_") or name == "name":
            continue
        if value is None or callable(value):
            continue
        if isinstance(value, (pd.DataFrame, pd.Series, dict, list, tuple, str, int, float, bool)):
            items[name] = value
    return items


def _select_pyaesa_extension_items(
    *,
    ext: Any,
    ext_name: str,
    include_attrs: Sequence[str],
    required_attrs: Sequence[str],
) -> Dict[str, Any]:
    """Return filtered extension attributes and enforce required fields."""
    cleaned_include = [str(name).strip() for name in include_attrs if str(name).strip()]
    cleaned_required = [str(name).strip() for name in required_attrs if str(name).strip()]
    selected: Dict[str, Any] = {}
    for attr_name in cleaned_include:
        value = getattr(ext, attr_name, None)
        if value is None or callable(value):
            continue
        if not isinstance(
            value, (pd.DataFrame, pd.Series, dict, list, tuple, str, int, float, bool)
        ):
            raise ValueError(
                f"Cannot persist extension '{ext_name}.{attr_name}': unsupported payload "
                f"type {type(value).__name__}."
            )
        selected[attr_name] = value

    missing = [name for name in cleaned_required if name not in selected]
    if missing:
        raise ValueError(
            f"Cannot persist extension '{ext_name}': required extension attributes "
            f"are missing {missing}."
        )
    return selected


def save_pyaesa_extension_pickles(
    *,
    iosys: pymrio.IOSystem,
    saved_dir: Path,
    lcia_methods: Sequence[str] | None,
) -> Dict[str, list[str]]:
    """Persist root extension payloads needed to recompute enacting metrics.

    The saved payload is explicit and deterministic:
    - each LCIA method keeps characterized post clip intermediates used by
      UNCASExt LCIA computations: ``S``, ``M``, and ``unit``.
    """
    summary: Dict[str, list[str]] = {}

    extension_names = [
        str(lcia_method).strip() for lcia_method in (lcia_methods or []) if str(lcia_method).strip()
    ]
    if not extension_names:
        return summary

    extensions_root = ensure_dir(_get_root_extensions_dir(saved_dir))
    for ext_name in extension_names:
        ext = getattr(iosys, ext_name, None)
        if ext is None:
            raise ValueError(f"Cannot persist extension '{ext_name}': missing on IOSystem.")
        payload_items = _select_pyaesa_extension_items(
            ext=ext,
            ext_name=ext_name,
            include_attrs=("S", "M", "unit"),
            required_attrs=("S", "M", "unit"),
        )
        ext_dir = extensions_root / ext_name
        ext_dir = ensure_dir(ext_dir)
        saved_names: list[str] = []
        for attr_name, payload in sorted(payload_items.items()):
            out_path = ext_dir / f"{attr_name}.pickle"
            with out_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            saved_names.append(attr_name)
        summary[ext_name] = saved_names
    return summary


def save_pymrio_calc_all_extensions(
    *,
    iosys: pymrio.IOSystem,
    saved_dir: Path,
    include_extensions: Sequence[str] | None = None,
) -> Dict[str, list[str]]:
    """Persist selected extension attributes from a calc_all-expanded IOSystem.

    Extension folder names are derived from extension instance names
    reported by ``iosys.get_extensions(instance_names=True)``.
    """
    get_extensions = getattr(iosys, "get_extensions", None)
    if not callable(get_extensions):
        raise ValueError("Cannot persist calc_all extensions: IOSystem.get_extensions is missing.")

    instance_names = list(cast(Iterable[Any], get_extensions(instance_names=True)))
    ext_instances = list(cast(Iterable[Any], get_extensions(data=True)))
    if len(instance_names) != len(ext_instances):
        raise ValueError(
            "Cannot persist calc_all extensions: instance names and extension "
            f"objects count mismatch ({len(instance_names)} vs {len(ext_instances)})."
        )
    include_set: set[str] | None = None
    if include_extensions is not None:
        include_set = {str(name).strip() for name in include_extensions if str(name).strip()}
    summary: Dict[str, list[str]] = {}
    extensions_root: Path | None = None
    for idx, ext in enumerate(ext_instances):
        ext_name = str(instance_names[idx]).strip()
        if not ext_name:
            raise ValueError(
                "Cannot persist calc_all extensions: blank extension instance "
                f"name at position {idx}."
            )
        if include_set is not None and ext_name not in include_set:
            continue
        payload_items = _extension_serializable_items(ext)
        if not payload_items:
            continue
        if extensions_root is None:
            extensions_root = ensure_dir(_get_preclip_extensions_dir(saved_dir))
        ext_dir = extensions_root / ext_name
        ext_dir = ensure_dir(ext_dir)
        saved_names: list[str] = []
        for attr_name, payload in sorted(payload_items.items()):
            out_path = ext_dir / f"{attr_name}.pickle"
            with out_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            saved_names.append(attr_name)
        summary[ext_name] = saved_names

    marker_path = _get_preclip_calc_all_marker_path(saved_dir)
    marker_path = ensure_file_parent(marker_path)
    marker_path.write_text(
        json.dumps(
            {
                "pymrio_calc_all": True,
                "extensions": sorted(summary.keys()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def preclip_calc_all_outputs_exist(saved_dir: Path) -> bool:
    """Return True when calc_all payload marker is present in preclip outputs."""
    return _get_preclip_calc_all_marker_path(saved_dir).exists()


def summarize_saved(saved_dir: Path) -> Dict[str, Any]:
    """Return root core matrices and root extension matrices for one year."""
    core = sorted(path.stem for path in saved_dir.glob("*.pickle"))
    extensions: Dict[str, list[str]] = {}
    extensions_root = _get_root_extensions_dir(saved_dir)
    if extensions_root.exists():
        for subdir in sorted(path for path in extensions_root.iterdir() if path.is_dir()):
            matrices = sorted(child.stem for child in subdir.glob("*.pickle"))
            if matrices:
                extensions[subdir.name] = matrices
    return {"core": core, "extensions": extensions}


def summarize_preclip_core(saved_dir: Path) -> list[str]:
    """Return preclip core matrix names stored directly under ``preclip/``."""
    preclip_dir = _get_preclip_dir(saved_dir)
    return sorted(path.stem for path in preclip_dir.glob("*.pickle"))


def prune_saved_dir(saved_dir: Path, keep_dirs: Sequence[str]) -> None:
    """Delete all content under ``saved_dir`` except directory names in ``keep_dirs``."""
    keep_set = {str(name) for name in keep_dirs}
    for child in saved_dir.iterdir():
        if child.is_dir():
            if child.name in keep_set:
                continue
            shutil.rmtree(child)
            continue
        child.unlink()


def save_lcia_fy_pickles(
    *,
    iosys: pymrio.IOSystem,
    saved_dir: Path,
    lcia_method_names: Sequence[str],
) -> list[str]:
    """Persist LCIA ``F_Y`` by method under enacting metrics level 1."""
    kept_methods: list[str] = []
    for lcia_method in lcia_method_names:
        lcia_method = str(lcia_method).strip()
        if not lcia_method:
            continue
        lcia_ext = getattr(iosys, lcia_method, None)
        if lcia_ext is None:
            continue
        f_y = getattr(lcia_ext, "F_Y", None)
        if f_y is None:
            continue
        out_dir = saved_dir / "enacting_metrics" / "level_1" / lcia_method
        out_dir = ensure_dir(out_dir)
        with (out_dir / "F_Y.pickle").open("wb") as handle:
            pickle.dump(f_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
        kept_methods.append(lcia_method)
    return kept_methods


def pyaesa_outputs_exist(
    saved_dir: Path,
    *,
    is_exio_source: bool,
    lcia_methods: Optional[Sequence[str]],
) -> bool:
    """Return True when expected UNCASExt output pickles are present."""
    util_dir = saved_dir / "utility_propag_uncasext"
    util_files = [
        util_dir / "x_to_rc.pickle",
        util_dir / "kappa.pickle",
        util_dir / "omega_reg.pickle",
    ]
    base = saved_dir / "enacting_metrics"
    l1_dir = base / "level_1"
    l2_dir = base / "level_2"
    units_file = base / "units.json"
    l1_files = [l1_dir / "fd_rf.pickle", l1_dir / "gva_rp.pickle"]
    l2_files = [
        l2_dir / "fd_rp_sp_rf.pickle",
        l2_dir / "fd_rp_sp.pickle",
        l2_dir / "fd_rf_sp.pickle",
        l2_dir / "gva_rp_sp.pickle",
    ]
    if is_exio_source and lcia_methods:
        cleaned_methods = [
            str(lcia_method).strip() for lcia_method in lcia_methods if str(lcia_method).strip()
        ]
        for lcia_method in cleaned_methods:
            l1_files.append(l1_dir / lcia_method / "F_Y.pickle")
            l1_files.extend(
                [
                    l1_dir / lcia_method / "e_cba_fd_reg.pickle",
                    l1_dir / lcia_method / "e_pba_reg.pickle",
                ]
            )
            l2_files.extend(
                [
                    l2_dir / lcia_method / "e_pba_rp_sp.pickle",
                    l2_dir / lcia_method / "e_cba_fd_rp_sp.pickle",
                    l2_dir / lcia_method / "e_cba_fd_rp_sp_rf.pickle",
                    l2_dir / lcia_method / "e_cba_td_rp_sp_rc.pickle",
                    l2_dir / lcia_method / "e_cba_td_rp_sp.pickle",
                    l2_dir / lcia_method / "e_cba_fd_rf_sp.pickle",
                    l2_dir / lcia_method / "e_cba_td_rc_sp.pickle",
                ]
            )
    return all(path.exists() for path in util_files + l1_files + l2_files + [units_file])
