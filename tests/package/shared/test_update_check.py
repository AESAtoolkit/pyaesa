import io
from importlib import metadata

import pytest
import requests

import pyaesa
from pyaesa.shared.runtime import update_check


class _Response:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> object:
        return self.payload


def _pypi_payload(*versions: tuple[str, bool]) -> dict[str, dict[str, list[dict[str, bool]]]]:
    return {"releases": {version: [{"yanked": yanked}] for version, yanked in versions}}


def _github_release(tag: str, body: str, **flags: bool) -> dict[str, object]:
    return {"tag_name": tag, "body": body, "draft": False, "prerelease": False, **flags}


def _request_sequence(responses: list[_Response]):
    calls: list[tuple[str, dict[str, object] | None]] = []

    def request_get(url: str, *, params=None, timeout=None):
        calls.append((url, params))
        response = responses.pop(0)
        return response

    return request_get, calls


@pytest.fixture(autouse=True)
def _enable_update_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYAESA_DISABLE_UPDATE_CHECK", raising=False)


def test_import_does_not_call_update_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_request(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(update_check.requests, "get", fail_request)
    assert pyaesa.__version__ == metadata.version("pyaesa")
    assert called is False


def test_imported_version_matches_installed_distribution() -> None:
    assert pyaesa.__version__ == metadata.version("pyaesa")


def test_latest_pypi_ignores_prerelease_and_yanked_files() -> None:
    payload = _pypi_payload(("1.2.6", False), ("1.2.7", True), ("1.3.0rc1", False))
    assert update_check._latest_stable_pypi_version(payload) == update_check.Version("1.2.6")


def test_malformed_pypi_file_entry_is_controlled_failure() -> None:
    with pytest.raises(update_check.UpdateCheckError):
        update_check._latest_stable_pypi_version({"releases": {"1.2.8": ["bad"]}})


def test_yanked_release_with_one_valid_file_is_usable() -> None:
    payload = {"releases": {"1.2.8": [{"yanked": True}, {"yanked": False}]}}
    assert update_check._latest_stable_pypi_version(payload) == update_check.Version("1.2.8")


def test_update_message_contains_sorted_cumulative_release_bodies() -> None:
    request_get, calls = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.6", False), ("1.2.8", False))),
            _Response(
                [
                    _github_release("v1.2.8", "eighth\n- fix"),
                    _github_release("v1.2.6", "sixth"),
                    _github_release("v1.2.7", "seventh"),
                    _github_release("v1.2.9", "future"),
                    _github_release("v1.2.5", "old"),
                    _github_release("v1.2.8rc1", "pre", prerelease=True),
                ]
            ),
            _Response([]),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    output = stream.getvalue()
    assert output.index("v1.2.6") < output.index("v1.2.7") < output.index("v1.2.8")
    assert "sixth" in output and "seventh" in output and "eighth\n- fix" in output
    assert calls[0][1] is None and calls[1][1] == {"per_page": 100, "page": 1}
    assert calls[2][1] == {"per_page": 100, "page": 2}


def test_current_or_newer_version_is_silent_and_once_only() -> None:
    request_get, calls = _request_sequence([_Response(_pypi_payload(("1.2.8", False)))])
    stream = io.StringIO()
    notifier = update_check.UpdateNotifier()
    notifier.check(installed_version="1.2.8", request_get=request_get, stream=stream)
    notifier.check(installed_version="1.2.7", request_get=request_get, stream=stream)
    assert stream.getvalue() == ""
    assert len(calls) == 1


def test_disabled_notifier_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYAESA_DISABLE_UPDATE_CHECK", "1")
    request_get, calls = _request_sequence([])
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=io.StringIO()
    )
    assert calls == []


@pytest.mark.parametrize(
    "payload",
    [None, {"bad": {}}, {"releases": {"1.2.8": "bad"}}, {"releases": {"invalid": [{}]}}],
)
def test_malformed_pypi_data_is_informational(payload: object) -> None:
    request_get, _ = _request_sequence([_Response(payload)])
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "could not be retrieved from PyPI" in stream.getvalue()


def test_pypi_failure_does_not_raise() -> None:
    request_get, _ = _request_sequence([_Response({}, requests.Timeout("network"))])
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "could not be retrieved from PyPI" in stream.getvalue()


def test_github_failure_keeps_confirmed_update() -> None:
    request_get, _ = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.8", False))),
            _Response([], requests.Timeout("network")),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    output = stream.getvalue()
    assert "pyaesa update available: 1.2.5 -> 1.2.8" in output
    assert "Release notes could not be retrieved" in output


def test_github_malformed_release_is_controlled_failure() -> None:
    request_get, _ = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.8", False))),
            _Response([{"bad": True}]),
            _Response([]),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "Release notes could not be retrieved" in stream.getvalue()


def test_github_invalid_version_is_controlled_failure() -> None:
    request_get, _ = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.8", False))),
            _Response([_github_release("not-a-version", "bad")]),
            _Response([]),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "Release notes could not be retrieved" in stream.getvalue()


def test_github_page_with_non_mapping_item_is_controlled_failure() -> None:
    request_get, _ = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.8", False))),
            _Response(["bad"]),
            _Response([]),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "Release notes could not be retrieved" in stream.getvalue()


def test_github_non_list_page_is_controlled_failure() -> None:
    request_get, _ = _request_sequence(
        [
            _Response(_pypi_payload(("1.2.8", False))),
            _Response({"unexpected": "payload"}),
        ]
    )
    stream = io.StringIO()
    update_check.UpdateNotifier().check(
        installed_version="1.2.5", request_get=request_get, stream=stream
    )
    assert "Release notes could not be retrieved" in stream.getvalue()


def test_package_api_resolution_invokes_checker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pyaesa,
        "check_for_updates",
        lambda *, installed_version: calls.append(installed_version),
    )
    pyaesa.__dict__.pop("set_workspace", None)
    pyaesa.__getattr__("set_workspace")
    pyaesa.__dict__.pop("set_workspace", None)
    pyaesa.__getattr__("set_workspace")
    assert calls == [pyaesa.__version__, pyaesa.__version__]


def test_package_level_checker_delegates_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYAESA_DISABLE_UPDATE_CHECK", "1")
    update_check.check_for_updates(installed_version="1.2.8")


def test_package_directory_lists_public_api() -> None:
    assert set(pyaesa.__all__).issubset(pyaesa.__dir__())
