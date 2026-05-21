import io

from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter


class _TTYStream(io.StringIO):
    def __init__(self, *, is_tty: bool) -> None:
        super().__init__()
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_transient_status_printer_show_and_finish() -> None:
    """TransientStatusPrinter delegates to YearProgressPrinter correctly."""
    stream = _TTYStream(is_tty=False)
    printer = TransientStatusPrinter(label="test-label")
    # Swap the internal printer's stream so we can capture output without a tty.
    printer._printer.stream = stream

    printer.show("step one")
    output = stream.getvalue()
    assert "step one" in output

    printer.show("step two")
    output = stream.getvalue()
    assert "step two" in output

    long_message = "stage " + "x" * 140
    printer.show(long_message)
    compacted = stream.getvalue().split("\r")[-1].strip()
    assert compacted.endswith("...")
    assert len(compacted) <= 100

    printer.finish()
