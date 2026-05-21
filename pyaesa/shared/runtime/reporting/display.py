"""Abstract terminal and notebook display capabilities for live status output.

These helpers isolate environment dependent rendering support such as TTY
refresh, ANSI multiline rerender, and IPython display handles so runtime status
printers can adapt without embedding frontend detection logic everywhere.
"""

import os
import sys
from collections.abc import Mapping
from html import escape
from importlib import import_module
from typing import Any, Callable

DEFAULT_IPY_SUPPORT = object()


def load_ipython_support(
    *,
    import_module_func: Callable[[str], Any] = import_module,
) -> tuple[Callable[[], Any | None] | None, type[Any] | None, type[Any] | None]:
    """Return optional IPython display helpers when IPython is installed."""
    get_ipython_func: Callable[[], Any | None] | None = None
    display_handle_cls: type[Any] | None = None
    html_cls: type[Any] | None = None

    try:
        get_ipython_candidate = getattr(
            import_module_func("IPython.core.getipython"),
            "get_ipython",
            None,
        )
    except (ImportError, ModuleNotFoundError):
        get_ipython_candidate = None
    if callable(get_ipython_candidate):
        get_ipython_func = get_ipython_candidate

    try:
        display_module = import_module_func("IPython.display")
        display_handle_candidate = getattr(display_module, "DisplayHandle", None)
        html_candidate = getattr(display_module, "HTML", None)
    except (ImportError, ModuleNotFoundError):
        display_handle_candidate = None
        html_candidate = None
    if display_handle_candidate is not None:
        display_handle_cls = display_handle_candidate
    if html_candidate is not None:
        html_cls = html_candidate

    return get_ipython_func, display_handle_cls, html_cls


def default_ipython_support() -> tuple[
    Callable[[], Any | None] | None,
    type[Any] | None,
    type[Any] | None,
]:
    """Return optional IPython display helpers without retaining package state."""
    return load_ipython_support()


def format_hms(seconds: float | None) -> str:
    """Format seconds as ``HH:MM:SS`` or a placeholder when unknown."""
    if seconds is None:
        return "--:--:--"
    whole = max(0, int(round(float(seconds))))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_hms_or_na(seconds: float | None) -> str:
    """Format seconds as ``HH:MM:SS`` or ``n/a`` when unavailable."""
    if seconds is None:
        return "n/a"
    return format_hms(seconds)


def short_source(source: str, *, max_len: int = 18) -> str:
    """Return a compact source label for progress displays."""
    text = str(source).strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return f"{text[: max_len - 3]}..."


def supports_live_refresh(stream: Any | None = None) -> bool:
    """Return whether stdout supports in place live progress refresh."""
    target_stream = sys.stdout if stream is None else stream
    try:
        return bool(target_stream.isatty())
    except (AttributeError, OSError):
        return False


def supports_multiline_rerender(
    *,
    stream: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether ANSI cursor rerender is safe for two line rendering."""
    target_env = os.environ if env is None else env
    if not supports_live_refresh(stream):
        return False
    if target_env.get("JPY_PARENT_PID"):
        return False
    term = str(target_env.get("TERM", "")).strip().lower()
    if term in {"", "dumb"}:
        return False
    return True


def supports_ipy_display(
    *,
    get_ipython_func: Callable[[], Any | None] | None | object = DEFAULT_IPY_SUPPORT,
    display_handle_cls: type[Any] | None | object = DEFAULT_IPY_SUPPORT,
) -> bool:
    """Return whether notebook display handle live updates are available."""
    if get_ipython_func is DEFAULT_IPY_SUPPORT or display_handle_cls is DEFAULT_IPY_SUPPORT:
        default_get_ipython, default_display_handle, _default_html = default_ipython_support()
    else:
        default_get_ipython = None
        default_display_handle = None
    effective_get_ipython = (
        default_get_ipython if get_ipython_func is DEFAULT_IPY_SUPPORT else get_ipython_func
    )
    effective_display_handle = (
        default_display_handle if display_handle_cls is DEFAULT_IPY_SUPPORT else display_handle_cls
    )
    if not callable(effective_get_ipython) or effective_display_handle is None:
        return False
    shell = effective_get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def write_to_stream(stream: Any, text: str, *, end: str = "") -> bool:
    """Write progress text to ``stream`` when the stream accepts status output.

    Progress rendering is diagnostic output only. A closed or invalid terminal
    stream must not fail the scientific calculation that is reporting progress.

    Returns:
        ``True`` when the text was written and flushed successfully, otherwise
        ``False``.
    """
    try:
        stream.write(f"{text}{end}")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    except (OSError, ValueError):
        return False
    return True


def render_ipy_text(
    *,
    text: str,
    handle: Any | None,
    display_handle_cls: type[Any] | None,
    html_cls: type[Any] | None,
) -> tuple[Any | None, bool]:
    """Render one notebook text block and return the updated handle."""
    if display_handle_cls is None or html_cls is None:
        return handle, False
    try:
        html_text = html_cls(f"<pre>{escape(text)}</pre>")
        if handle is None:
            new_handle = display_handle_cls()
            new_handle.display(html_text)
            return new_handle, True
        handle.update(html_text)
        return handle, True
    except RuntimeError:
        return handle, False
