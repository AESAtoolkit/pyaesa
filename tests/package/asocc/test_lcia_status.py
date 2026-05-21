from pyaesa.asocc.data.lcia_status import (
    format_lcia_missing_reason,
    resolve_lcia_status,
)


def test_format_lcia_missing_reason_normalizes_values() -> None:
    assert format_lcia_missing_reason(None) is None
    assert format_lcia_missing_reason("") is None
    assert format_lcia_missing_reason("missing factor")
    single_reason = format_lcia_missing_reason(["land"])
    assert single_reason is not None
    assert "land" in single_reason
    combined_reason = format_lcia_missing_reason(["land", "climate"])
    assert combined_reason is not None
    assert "land" in combined_reason
    assert "climate" in combined_reason
    assert format_lcia_missing_reason(["", "   "]) is None


def test_resolve_lcia_status_handles_available_and_unavailable() -> None:
    unavailable_entry = {
        "lcia_status": {
            "pb_lcia": {
                "available": False,
                "missing": ["land"],
            }
        }
    }
    available_entry = {
        "lcia_status": {
            "pb_lcia": {
                "available": True,
            }
        }
    }

    assert resolve_lcia_status(unavailable_entry, "pb_lcia") == (
        False,
        format_lcia_missing_reason(["land"]),
    )
    assert resolve_lcia_status(available_entry, "pb_lcia") == (True, None)
