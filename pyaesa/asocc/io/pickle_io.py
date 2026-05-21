"""Pickle loading for allocation and IO-LCA inputs."""

import pickle
import platform
import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _read_metadata,
)

_PICKLE_READ_EXCEPTIONS = (
    pickle.UnpicklingError,
    EOFError,
    AttributeError,
    ValueError,
    TypeError,
    ImportError,
    ModuleNotFoundError,
)


class _CompatUnpickler(pickle.Unpickler):
    """Read package cache files across supported NumPy environments."""

    _MODULE_REMAP = {
        "numpy._core.numeric": "numpy",
        "numpy.core.numeric": "numpy",
    }

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self._MODULE_REMAP.get(module, module), name)


def _current_env_versions() -> dict[str, str]:
    """Return compact runtime version info for diagnostics."""
    env: dict[str, str] = {"python": platform.python_version()}
    for package_name in ("numpy", "pandas", "scipy", "pymrio", "pyarrow"):
        env[package_name] = importlib_metadata.version(package_name)
    return env


def _format_env_versions(env: dict[str, str] | None) -> str:
    """Format environment versions in stable key order."""
    if not env:
        return "unknown"
    keys = ["python", "numpy", "pandas", "scipy", "pymrio", "pyarrow"]
    ordered = [f"{key}={env[key]}" for key in keys if key in env]
    ordered.extend(f"{key}={env[key]}" for key in sorted(set(env) - set(keys)))
    return ", ".join(ordered) if ordered else "unknown"


def _infer_saved_env_for_mrio_pickle(path: Path) -> dict[str, str] | None:
    """Try to resolve stored runtime env from MRIO metadata for one pickle path."""
    year_dir: Path | None = None
    version_dir: Path | None = None
    source_dir: Path | None = None
    for candidate in path.parents:
        parent = candidate.parent
        if (parent / "metadata.json").exists():
            year_dir = candidate
            version_dir = parent
            source_dir = parent.parent
            break
    if year_dir is None or version_dir is None or source_dir is None:
        return None
    year_tokens = re.findall(r"(19\d{2}|20\d{2}|21\d{2})", year_dir.name)
    if not year_tokens:
        return None
    year = int(year_tokens[-1])
    version_tag = str(version_dir.name).strip()
    matrix_version = (
        None
        if version_tag == "original_classification"
        else (
            version_tag[len("custom_classification_") :]
            if version_tag.startswith("custom_classification_")
            else version_tag
        )
    )
    try:
        metadata = _read_metadata(str(source_dir.name).strip(), matrix_version=matrix_version)
        year_entry = _get_year_entry(metadata, year)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
        ModuleNotFoundError,
    ):
        return None
    runtime_env = year_entry.get("runtime_env") if isinstance(year_entry, dict) else None
    if not isinstance(runtime_env, dict):
        return None
    return {str(key): str(value) for key, value in runtime_env.items()}


def _pickle_env_error(path: Path) -> RuntimeError:
    """Build a clear pickle compatibility error with remediation steps."""
    return RuntimeError(
        f"Failed to load pickle artifact at {path}. "
        f"Saved env: {_format_env_versions(_infer_saved_env_for_mrio_pickle(path))}. "
        f"Current env: {_format_env_versions(_current_env_versions())}. "
        "Reprocess MRIO in the current environment or run deterministic_asocc in the "
        "same environment that produced these MRIO pickle outputs."
    )


def read_pickle(path: Path) -> Any:
    """Load one pickle from disk with environment compatibility guidance."""
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError:
            handle.seek(0)
            try:
                return _CompatUnpickler(handle).load()
            except _PICKLE_READ_EXCEPTIONS as fallback_exc:
                raise _pickle_env_error(path) from fallback_exc
        except _PICKLE_READ_EXCEPTIONS as exc:
            raise _pickle_env_error(path) from exc
