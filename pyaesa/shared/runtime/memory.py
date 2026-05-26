"""Shared runtime memory budgeting helpers."""

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass

BYTES_PER_GIB = 1024**3
# Usable package work budget when memory detection is unavailable.
FALLBACK_RUNTIME_BUDGET_BYTES = 4 * BYTES_PER_GIB
# Share of currently available RAM reserved outside planned package work, with a 1 GiB floor.
MEMORY_OS_RESERVE_PERCENT = 10
# Planned package work cap as a share of total physical RAM.
MEMORY_PHYSICAL_LIMIT_PERCENT = 75
# Planned package work cap as a share of currently available RAM.
MEMORY_AVAILABLE_LIMIT_PERCENT = 90


@dataclass(frozen=True)
class RuntimeMemoryBudget:
    """Call scoped runtime memory budget derived from system memory."""

    physical_memory_bytes: int
    available_memory_bytes: int
    operating_system_reserve_bytes: int
    minimal_working_block_bytes: int
    budget_bytes: int


def runtime_memory_budget(
    *,
    minimal_working_block_bytes: int = 1,
    physical_memory_bytes: int | None = None,
    available_memory_bytes: int | None = None,
) -> RuntimeMemoryBudget:
    """Return the shared runtime memory budget for one public package call.

    If platform memory detection or current available RAM cannot support one
    minimal block, the fallback profile yields at least 4 GiB of usable
    package work budget after reserve and cap policy.
    """
    minimal = int(minimal_working_block_bytes)
    if physical_memory_bytes is None or available_memory_bytes is None:
        detected_physical, detected_available = _detect_system_memory_bytes()
        physical = detected_physical if physical_memory_bytes is None else physical_memory_bytes
        available = detected_available if available_memory_bytes is None else available_memory_bytes
    else:
        physical = physical_memory_bytes
        available = available_memory_bytes
    physical = int(physical)
    available = int(available)
    reserve = max(BYTES_PER_GIB, _percent_of(available, MEMORY_OS_RESERVE_PERCENT))
    available_after_reserve = available - reserve
    if available_after_reserve < minimal:
        physical, available = _fallback_system_memory_bytes(
            minimal_working_block_bytes=minimal,
        )
        reserve = max(BYTES_PER_GIB, _percent_of(available, MEMORY_OS_RESERVE_PERCENT))
        available_after_reserve = available - reserve
    candidate = min(
        _percent_of(physical, MEMORY_PHYSICAL_LIMIT_PERCENT),
        _percent_of(available, MEMORY_AVAILABLE_LIMIT_PERCENT),
    )
    budget = max(min(candidate, available_after_reserve), minimal)
    return RuntimeMemoryBudget(
        physical_memory_bytes=physical,
        available_memory_bytes=available,
        operating_system_reserve_bytes=reserve,
        minimal_working_block_bytes=minimal,
        budget_bytes=int(budget),
    )


def memory_bounded_rows(
    *,
    bytes_per_row: int,
    working_arrays: int = 1,
    minimum_rows: int = 1,
    memory_budget_bytes: int | None = None,
) -> int:
    """Return a row count derived from the shared runtime budget."""
    row_size = int(bytes_per_row)
    arrays = int(working_arrays)
    minimum = int(minimum_rows)
    retained_row_size = row_size * arrays
    budget = runtime_working_budget_bytes(
        memory_budget_bytes=memory_budget_bytes,
        minimal_working_block_bytes=retained_row_size * minimum,
    )
    return max(minimum, int(budget) // retained_row_size)


def runtime_working_budget_bytes(
    *,
    memory_budget_bytes: int | None,
    minimal_working_block_bytes: int,
) -> int:
    """Return runtime budget bytes after accounting for current process RSS."""
    if memory_budget_bytes is not None:
        return int(memory_budget_bytes)
    budget = runtime_memory_budget(
        minimal_working_block_bytes=minimal_working_block_bytes
    ).budget_bytes
    current_rss = current_process_rss_bytes()
    if current_rss <= 0:
        return budget
    working_budget = int(budget) - int(current_rss)
    if working_budget < int(minimal_working_block_bytes):
        return int(minimal_working_block_bytes)
    return working_budget


def current_process_rss_bytes() -> int:
    """Return current process resident memory bytes when available."""
    if os.name == "nt":
        return _current_windows_process_rss_bytes()
    return _current_posix_process_rss_bytes()


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _current_windows_process_rss_bytes() -> int:
    counters = _WindowsProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_WindowsProcessMemoryCounters)
    win_dll = getattr(ctypes, "WinDLL")
    psapi = win_dll("psapi", use_last_error=True)
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_WindowsProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not bool(psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)):
        return 0
    return int(counters.WorkingSetSize)


def _current_posix_process_rss_bytes() -> int:
    page_size = _sysconf_int("SC_PAGE_SIZE")
    statm_path = "/proc/self/statm"
    if page_size is not None and os.path.exists(statm_path):
        with open(statm_path, encoding="utf-8") as handle:
            parts = handle.read().split()
        if len(parts) >= 2:
            return int(parts[1]) * int(page_size)
    return 0


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _detect_system_memory_bytes() -> tuple[int, int]:
    if os.name == "nt":
        return _detect_windows_memory_bytes()
    return _detect_posix_memory_bytes()


def _detect_windows_memory_bytes() -> tuple[int, int]:
    status = _WindowsMemoryStatus()
    status.dwLength = ctypes.sizeof(_WindowsMemoryStatus)
    win_dll = getattr(ctypes, "WinDLL")
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_WindowsMemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not bool(kernel32.GlobalMemoryStatusEx(ctypes.byref(status))):
        return _fallback_system_memory_bytes()
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _detect_posix_memory_bytes() -> tuple[int, int]:
    page_size = _sysconf_int("SC_PAGE_SIZE")
    physical_pages = _sysconf_int("SC_PHYS_PAGES")
    available_pages = _sysconf_int("SC_AVPHYS_PAGES")
    if page_size is None or physical_pages is None:
        return _fallback_system_memory_bytes()
    if available_pages is None:
        available_pages = physical_pages
    return int(page_size) * int(physical_pages), int(page_size) * int(available_pages)


def _fallback_system_memory_bytes(*, minimal_working_block_bytes: int = 1) -> tuple[int, int]:
    budget = max(FALLBACK_RUNTIME_BUDGET_BYTES, int(minimal_working_block_bytes))
    physical = _ceil_div(budget * 100, MEMORY_PHYSICAL_LIMIT_PERCENT)
    reserve = max(BYTES_PER_GIB, _percent_of(physical, MEMORY_OS_RESERVE_PERCENT))
    available = max(
        _ceil_div(budget * 100, MEMORY_AVAILABLE_LIMIT_PERCENT),
        budget + reserve,
    )
    return physical, available


def _percent_of(value: int, percent: int) -> int:
    return int(value) * int(percent) // 100


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


def _sysconf_int(name: str) -> int | None:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        value = sysconf(name)
    except (OSError, ValueError):
        return None
    return int(value) if int(value) > 0 else None
