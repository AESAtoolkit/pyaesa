import builtins
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pyaesa.shared.runtime import memory as memory_mod
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import read_run_interval_index
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows, SparseRunRowsWriter


class _Callable:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.argtypes: list[Any] = []
        self.restype: Any = None

    def __call__(self, *_args: Any) -> Any:
        if callable(self.result):
            return self.result(*_args)
        return self.result


class _ProcessKernel32:
    GetCurrentProcess = _Callable(1)


class _FailingPsapi:
    GetProcessMemoryInfo = _Callable(0)


class _FailingMemoryKernel32:
    GlobalMemoryStatusEx = _Callable(0)


class _SuccessfulPsapi:
    def __init__(self, working_set_size: int) -> None:
        self.working_set_size = working_set_size
        self.GetProcessMemoryInfo = _Callable(self._write_process_memory)

    def _write_process_memory(self, _handle: object, counters_pointer: object, _size: int) -> int:
        cast(Any, counters_pointer)._obj.WorkingSetSize = self.working_set_size
        return 1


class _SuccessfulMemoryKernel32:
    GetCurrentProcess = _Callable(1)

    def __init__(self, *, total_phys: int, avail_phys: int) -> None:
        self.total_phys = total_phys
        self.avail_phys = avail_phys
        self.GlobalMemoryStatusEx = _Callable(self._write_memory_status)

    def _write_memory_status(self, status_pointer: object) -> int:
        cast(Any, status_pointer)._obj.ullTotalPhys = self.total_phys
        cast(Any, status_pointer)._obj.ullAvailPhys = self.avail_phys
        return 1


def _windows_memory_api(
    monkeypatch: pytest.MonkeyPatch, *, physical: int, available: int, rss: int
) -> None:
    """Supply Windows memory observations at the external API boundary."""
    kernel = _SuccessfulMemoryKernel32(total_phys=physical, avail_phys=available)

    def win_dll(name: str, *, use_last_error: bool) -> object:
        assert use_last_error is True
        return _SuccessfulPsapi(working_set_size=rss) if name == "psapi" else kernel

    monkeypatch.setattr(memory_mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(memory_mod.ctypes, "WinDLL", win_dll, raising=False)


@pytest.mark.parametrize(
    ("rss", "expected"),
    [(0, 7 * 1024**3), (12 * 1024**3, 7 * 1024**3), (23 * 1024**3, 1024**3), (24 * 1024**3, 24)],
)
def test_runtime_working_budget_counts_resident_memory_once(
    monkeypatch: pytest.MonkeyPatch, rss: int, expected: int
) -> None:
    _windows_memory_api(monkeypatch, physical=32 * 1024**3, available=8 * 1024**3, rss=rss)
    assert (
        memory_mod.runtime_working_budget_bytes(
            memory_budget_bytes=None,
            minimal_working_block_bytes=24,
        )
        == expected
    )


def test_allocated_sparse_batch_preserves_values_without_tiny_fragments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _windows_memory_api(monkeypatch, physical=32 * 1024**3, available=8 * 1024**3, rss=12 * 1024**3)
    rows = SparseRunRows(
        run_index=np.repeat(np.arange(3, dtype=np.int64), 16),
        public_row_id=np.tile(np.arange(16, dtype=np.int64), 3),
        values=np.arange(48, dtype=np.float64) / 7,
        value_column="acc",
    )
    path = tmp_path / "acc_runs.parquet"
    with SparseRunRowsWriter(path=path, output_format="parquet") as writer:
        writer.write_batch(rows=rows, batch_index=0)
    intervals = read_run_interval_index(path=path, output_format="parquet")
    assert intervals["row_count"].tolist() == [48]
    actual = pd.read_parquet(path)
    expected = pd.DataFrame(
        {"run_index": rows.run_index, "public_row_id": rows.public_row_id, "acc": rows.values}
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_windows_memory_helpers_fall_back_when_windows_api_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def process_win_dll(name: str, *, use_last_error: bool) -> object:
        assert use_last_error is True
        return _FailingPsapi() if name == "psapi" else _ProcessKernel32()

    monkeypatch.setattr(memory_mod.ctypes, "WinDLL", process_win_dll, raising=False)
    assert memory_mod._current_windows_process_rss_bytes() == 0

    def memory_win_dll(name: str, *, use_last_error: bool) -> object:
        assert name == "kernel32"
        assert use_last_error is True
        return _FailingMemoryKernel32()

    monkeypatch.setattr(memory_mod.ctypes, "WinDLL", memory_win_dll, raising=False)
    assert memory_mod._detect_windows_memory_bytes() == memory_mod._fallback_system_memory_bytes()


def test_windows_memory_helpers_read_successful_windows_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def process_win_dll(name: str, *, use_last_error: bool) -> object:
        assert use_last_error is True
        return _SuccessfulPsapi(working_set_size=1234) if name == "psapi" else _ProcessKernel32()

    monkeypatch.setattr(memory_mod.ctypes, "WinDLL", process_win_dll, raising=False)
    monkeypatch.setattr(memory_mod.os, "name", "nt")
    assert memory_mod.current_process_rss_bytes() == 1234

    def memory_win_dll(name: str, *, use_last_error: bool) -> object:
        assert name == "kernel32"
        assert use_last_error is True
        return _SuccessfulMemoryKernel32(total_phys=4096, avail_phys=2048)

    monkeypatch.setattr(memory_mod.ctypes, "WinDLL", memory_win_dll, raising=False)
    assert memory_mod._detect_system_memory_bytes() == (4096, 2048)


def test_posix_memory_helpers_use_proc_and_sysconf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_mod.os, "sysconf", lambda name: 4096, raising=False)
    monkeypatch.setattr(memory_mod.os.path, "exists", lambda path: path == "/proc/self/statm")
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *, encoding: StringIO("10 3\n"),
    )
    with monkeypatch.context() as platform:
        platform.setattr(memory_mod.os, "name", "posix")
        assert memory_mod.current_process_rss_bytes() == 12_288

    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *, encoding: StringIO("10\n"),
    )
    assert memory_mod._current_posix_process_rss_bytes() == 0

    monkeypatch.setattr(memory_mod.os.path, "exists", lambda path: False)
    assert memory_mod._current_posix_process_rss_bytes() == 0

    sysconf_values: dict[str, int | None] = {
        "SC_PAGE_SIZE": 4096,
        "SC_PHYS_PAGES": 100,
        "SC_AVPHYS_PAGES": 25,
    }
    monkeypatch.setattr(memory_mod.os, "sysconf", lambda name: sysconf_values[name], raising=False)
    with monkeypatch.context() as platform:
        platform.setattr(memory_mod.os, "name", "posix")
        assert memory_mod._detect_system_memory_bytes() == (409_600, 102_400)

    sysconf_values["SC_AVPHYS_PAGES"] = 0
    assert memory_mod._detect_posix_memory_bytes() == (409_600, 409_600)

    monkeypatch.setattr(memory_mod.os, "sysconf", lambda name: 0, raising=False)
    assert memory_mod._detect_posix_memory_bytes() == memory_mod._fallback_system_memory_bytes()


def test_sysconf_int_normalizes_unavailable_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory_mod.os, "sysconf", None, raising=False)
    assert memory_mod._sysconf_int("SC_PAGE_SIZE") is None

    def failing_sysconf(_name: str) -> int:
        raise ValueError("unknown")

    monkeypatch.setattr(memory_mod.os, "sysconf", failing_sysconf, raising=False)
    assert memory_mod._sysconf_int("SC_PAGE_SIZE") is None

    monkeypatch.setattr(memory_mod.os, "sysconf", lambda _name: 0, raising=False)
    assert memory_mod._sysconf_int("SC_PAGE_SIZE") is None

    monkeypatch.setattr(memory_mod.os, "sysconf", lambda _name: 4096, raising=False)
    assert memory_mod._sysconf_int("SC_PAGE_SIZE") == 4096
