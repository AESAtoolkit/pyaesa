from pyaesa.process.mrios.utils.pipeline.lcia_tracking import (
    expected_pyaesa_lcia_methods,
)
from pyaesa.process.mrios.utils.io.metadata import _metadata_satisfies


def test_expected_pyaesa_methods_uses_request_when_no_metadata() -> None:
    expected = expected_pyaesa_lcia_methods(
        year_meta=None,
        requested_methods=["pb_lcia"],
    )
    assert expected == ["pb_lcia"]


def test_expected_pyaesa_methods_respects_explicit_empty_metadata_list() -> None:
    expected = expected_pyaesa_lcia_methods(
        year_meta={"enacting_metrics": {"lcia_methods": []}},
        requested_methods=["pb_lcia"],
    )
    assert expected == ["pb_lcia"]


def test_expected_pyaesa_methods_uses_metadata_methods_without_status_filtering() -> None:
    expected = expected_pyaesa_lcia_methods(
        year_meta={
            "enacting_metrics": {"lcia_methods": ["pb_lcia", "gwp100_lcia"]},
            "lcia_status": {
                "pb_lcia": {"available": False, "missing": ["land"]},
                "gwp100_lcia": {"available": True},
            },
        },
        requested_methods=["pb_lcia", "gwp100_lcia"],
    )
    assert expected == ["pb_lcia", "gwp100_lcia"]


def test_expected_pyaesa_methods_merges_metadata_and_requested() -> None:
    expected = expected_pyaesa_lcia_methods(
        year_meta={"enacting_metrics": {"lcia_methods": ["pb_lcia"]}},
        requested_methods=["gwp100_lcia"],
    )
    assert expected == ["pb_lcia", "gwp100_lcia"]


def test_expected_pyaesa_methods_excludes_requested_unavailable_methods() -> None:
    expected = expected_pyaesa_lcia_methods(
        year_meta={
            "enacting_metrics": {"lcia_methods": ["gwp100_lcia"]},
            "lcia_status": {"pb_lcia": {"available": False, "missing": ["land"]}},
        },
        requested_methods=["pb_lcia"],
    )
    assert expected == ["gwp100_lcia"]


def test_metadata_satisfies_treats_unavailable_lcia_status_as_satisfied() -> None:
    entry = {
        "core": ["A"],
        "extensions": {
            "pb_lcia": {"available": False, "missing": ["land"]},
        },
        "lcia_status": {
            "pb_lcia": {"available": False, "missing": ["land"]},
        },
    }
    assert _metadata_satisfies(
        entry,
        saved_exists=True,
        required_core=["A"],
        required_extensions=[],
        required_lcia_method="pb_lcia",
    )


def test_metadata_satisfies_fails_when_required_lcia_entry_is_missing() -> None:
    entry = {
        "core": ["A"],
        "extensions": {},
    }
    assert not _metadata_satisfies(
        entry,
        saved_exists=True,
        required_core=["A"],
        required_extensions=[],
        required_lcia_method="pb_lcia",
    )
