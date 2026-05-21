import io
from types import SimpleNamespace

from pyaesa.shared.runtime.reporting import display as display_mod


class _TTYStream(io.StringIO):
    def __init__(self, *, is_tty: bool, fail_isatty: bool = False) -> None:
        super().__init__()
        self._is_tty = is_tty
        self._fail_isatty = fail_isatty

    def isatty(self) -> bool:
        if self._fail_isatty:
            raise OSError("isatty failed")
        return self._is_tty


class ZMQInteractiveShell:
    pass


class _TerminalShell:
    pass


class _NoFlushStream:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.flush = 1

    def write(self, text: str) -> None:
        self.parts.append(text)


class _FailingStream:
    def __init__(self, *, fail_on_flush: bool = False) -> None:
        self.parts: list[str] = []
        self._fail_on_flush = fail_on_flush

    def write(self, text: str) -> None:
        if not self._fail_on_flush:
            raise OSError("stream is unavailable")
        self.parts.append(text)

    def flush(self) -> None:
        if self._fail_on_flush:
            raise ValueError("stream is closed")


def test_load_ipython_support_and_format_helpers() -> None:
    def missing_import(_name: str):
        raise ModuleNotFoundError("IPython missing")

    get_ipython_func, display_handle_cls, html_cls = display_mod.load_ipython_support(
        import_module_func=missing_import,
    )
    assert get_ipython_func is None
    assert display_handle_cls is None
    assert html_cls is None

    fake_modules = {
        "IPython.core.getipython": SimpleNamespace(
            get_ipython=lambda: ZMQInteractiveShell(),
        ),
        "IPython.display": SimpleNamespace(DisplayHandle=object, HTML=object),
    }

    def fake_import(name: str):
        return fake_modules[name]

    get_ipython_func, display_handle_cls, html_cls = display_mod.load_ipython_support(
        import_module_func=fake_import,
    )
    assert get_ipython_func is not None
    assert display_handle_cls is object
    assert html_cls is object
    default_support = display_mod.default_ipython_support()
    assert len(default_support) == 3
    assert display_mod.default_ipython_support() is not default_support

    assert display_mod.format_hms(None) == "--:--:--"
    assert display_mod.format_hms(0) == "00:00:00"
    assert display_mod.format_hms(3661) == "01:01:01"
    assert display_mod.format_hms_or_na(None) == "n/a"
    assert display_mod.format_hms_or_na(1) == "00:00:01"
    assert display_mod.short_source("oecd") == "oecd"
    assert display_mod.short_source("abcdefghijklmnopqrstuvwxyz", max_len=6) == "abc..."
    assert display_mod.short_source("abcdef", max_len=3) == "abc"


def test_support_detectors_cover_terminal_and_notebook_cases() -> None:
    tty_stream = _TTYStream(is_tty=True)
    broken_stream = _TTYStream(is_tty=True, fail_isatty=True)

    assert display_mod.supports_live_refresh(tty_stream) is True
    assert display_mod.supports_live_refresh(broken_stream) is False

    assert display_mod.supports_multiline_rerender(stream=tty_stream, env={"TERM": "xterm"}) is True
    assert display_mod.supports_multiline_rerender(stream=tty_stream, env={"TERM": ""}) is False
    assert display_mod.supports_multiline_rerender(stream=tty_stream, env={"TERM": "dumb"}) is False
    assert (
        display_mod.supports_multiline_rerender(
            stream=tty_stream,
            env={"TERM": "xterm", "JPY_PARENT_PID": "1"},
        )
        is False
    )
    assert display_mod.supports_multiline_rerender(stream=_TTYStream(is_tty=False), env={}) is False

    assert (
        display_mod.supports_ipy_display(
            get_ipython_func=lambda: ZMQInteractiveShell(),
            display_handle_cls=object,
        )
        is True
    )
    assert (
        display_mod.supports_ipy_display(
            get_ipython_func=lambda: _TerminalShell(),
            display_handle_cls=object,
        )
        is False
    )
    assert (
        display_mod.supports_ipy_display(get_ipython_func=lambda: None, display_handle_cls=object)
        is False
    )
    assert (
        display_mod.supports_ipy_display(get_ipython_func=None, display_handle_cls=object) is False
    )
    assert (
        display_mod.supports_ipy_display(
            get_ipython_func=lambda: ZMQInteractiveShell(), display_handle_cls=None
        )
        is False
    )
    assert isinstance(display_mod.supports_ipy_display(), bool)

    stream = _NoFlushStream()
    assert display_mod.write_to_stream(stream, "hello", end="!") is True
    assert stream.parts == ["hello!"]

    assert display_mod.write_to_stream(_FailingStream(), "ignored") is False

    flush_stream = _FailingStream(fail_on_flush=True)
    assert display_mod.write_to_stream(flush_stream, "written") is False
    assert flush_stream.parts == ["written"]
