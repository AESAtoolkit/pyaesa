import io
import pickle
from pathlib import Path

import numpy as np
import pytest

from pyaesa.asocc.io import pickle_io as mod
from pyaesa.process.mrios.utils.io.metadata import _set_year_entry, _write_metadata
from pyaesa.process.mrios.utils.io.paths import _get_year_saved_path


def test_pickle_env_cover_metadata_inference(project_repo: Path) -> None:
    env = mod._current_env_versions()
    assert "python" in env
    assert mod._format_env_versions(None) == "unknown"
    assert mod._format_env_versions({}) == "unknown"
    assert mod._format_env_versions({"extra": "x", "numpy": "2.0", "python": "3.14"}) == (
        "python=3.14, numpy=2.0, extra=x"
    )

    orphan_path = project_repo / "orphan" / "artifact.pickle"
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    assert mod._infer_saved_env_for_mrio_pickle(orphan_path) is None

    payload = {
        "version_tag": "custom_classification_demo",
        "grouping": {},
        "labels": {},
        "years": {},
    }
    _set_year_entry(
        payload,
        2019,
        {
            "core": ["A"],
            "extensions": {},
            "runtime_env": {"python": "3.10", "numpy": 1.26},
        },
    )
    _write_metadata("oecd_v2025", payload, matrix_version="demo")

    saved_dir = _get_year_saved_path("oecd_v2025", 2019, matrix_version="demo")
    saved_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.parent.joinpath("metadata.json").write_text("{}", encoding="utf-8")
    pickle_path = saved_dir / "artifact.pickle"
    pickle_path.write_bytes(b"pickle")
    assert mod._infer_saved_env_for_mrio_pickle(pickle_path) == {
        "python": "3.10",
        "numpy": "1.26",
    }

    no_year_dir = (
        project_repo
        / "data_processed"
        / "mrio"
        / "oecd_v2025"
        / "custom_classification_demo"
        / "saved"
    )
    no_year_dir.mkdir(parents=True, exist_ok=True)
    no_year_dir.parent.joinpath("metadata.json").write_text("{}", encoding="utf-8")
    assert mod._infer_saved_env_for_mrio_pickle(no_year_dir / "artifact.pickle") is None

    invalid_runtime_payload = {
        "version_tag": "original_classification",
        "grouping": {},
        "labels": {},
        "years": {},
    }
    _set_year_entry(
        invalid_runtime_payload,
        2020,
        {
            "core": ["A"],
            "extensions": {},
            "runtime_env": ["bad"],
        },
    )
    _write_metadata("oecd_v2025", invalid_runtime_payload, matrix_version=None)
    original_dir = _get_year_saved_path("oecd_v2025", 2020, matrix_version=None)
    original_dir.mkdir(parents=True, exist_ok=True)
    original_dir.parent.joinpath("metadata.json").write_text("{}", encoding="utf-8")
    assert mod._infer_saved_env_for_mrio_pickle(original_dir / "artifact.pickle") is None

    unknown_dir = (
        project_repo
        / "data_processed"
        / "mrio"
        / "unknown_source"
        / "original_classification"
        / "saved_2019"
    )
    unknown_dir.mkdir(parents=True, exist_ok=True)
    unknown_dir.parent.joinpath("metadata.json").write_text("{}", encoding="utf-8")
    assert mod._infer_saved_env_for_mrio_pickle(unknown_dir / "artifact.pickle") is None


def test_pickle_reader_and_compat_unpickler_cover_success_and_error_paths(
    project_repo: Path,
) -> None:
    compat_payload = pickle.dumps(np.dtype("int64"), protocol=0).replace(
        b"cnumpy\n",
        b"cnumpy.core.numeric\n",
        1,
    )
    assert str(mod._CompatUnpickler(io.BytesIO(compat_payload)).load()) == "int64"

    ok_path = project_repo / "simple.pickle"
    ok_path.write_bytes(pickle.dumps({"answer": 42}))
    assert mod.read_pickle(ok_path) == {"answer": 42}

    payload = {"version_tag": "original_classification", "grouping": {}, "labels": {}, "years": {}}
    _set_year_entry(
        payload,
        2019,
        {
            "core": ["A"],
            "extensions": {},
            "runtime_env": {"python": "3.10"},
        },
    )
    _write_metadata("oecd_v2025", payload, matrix_version=None)

    saved_dir = _get_year_saved_path("oecd_v2025", 2019, matrix_version=None)
    saved_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.parent.joinpath("metadata.json").write_text("{}", encoding="utf-8")
    broken_path = saved_dir / "broken.pickle"
    broken_path.write_bytes(b"not a pickle")

    error = mod._pickle_env_error(broken_path)
    assert "3.10" in str(error)

    with pytest.raises(RuntimeError):
        mod.read_pickle(broken_path)

    missing_module_path = saved_dir / "missing_module.pickle"
    missing_module_path.write_bytes(
        compat_payload.replace(b"numpy.core.numeric", b"numpy._core.missing", 1)
    )
    with pytest.raises(RuntimeError):
        mod.read_pickle(missing_module_path)
