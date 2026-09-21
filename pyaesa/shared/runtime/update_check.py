"""Internal notification for newer pyaesa releases."""

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

import requests
from packaging.version import InvalidVersion, Version

_PYPI_URL = "https://pypi.org/pypi/pyaesa/json"
_GITHUB_URL = "https://api.github.com/repos/AESAtoolkit/pyaesa/releases"
_HISTORY_URL = "https://github.com/AESAtoolkit/pyaesa/releases"
_REQUEST_TIMEOUT = (1.0, 2.0)
_RequestGet = Callable[..., Any]


class UpdateCheckError(RuntimeError):
    """Expected failure while retrieving or validating update metadata."""


def _check_disabled() -> bool:
    return os.environ.get("PYAESA_DISABLE_UPDATE_CHECK") == "1"


def _get_json(*, request_get: _RequestGet, url: str, params: dict[str, int] | None = None) -> Any:
    try:
        response = request_get(url, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise UpdateCheckError from exc


def _stable_version(value: object) -> Version | None:
    try:
        version = Version(str(value))
    except (InvalidVersion, TypeError):
        return None
    if version.is_prerelease or version.is_devrelease:
        return None
    return version


def _latest_stable_pypi_version(payload: object) -> Version:
    if not isinstance(payload, Mapping):
        raise UpdateCheckError
    releases = payload.get("releases")
    if not isinstance(releases, Mapping):
        raise UpdateCheckError
    usable: list[Version] = []
    for raw_version, files in releases.items():
        version = _stable_version(raw_version)
        if version is None:
            continue
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            raise UpdateCheckError
        if any(isinstance(file, Mapping) and not bool(file.get("yanked", False)) for file in files):
            usable.append(version)
    if not usable:
        raise UpdateCheckError
    return max(usable)


def _fetch_pypi_version(*, request_get: _RequestGet) -> Version:
    return _latest_stable_pypi_version(_get_json(request_get=request_get, url=_PYPI_URL))


def _fetch_github_releases(*, request_get: _RequestGet) -> list[Mapping[str, object]]:
    releases: list[Mapping[str, object]] = []
    page = 1
    while True:
        payload = _get_json(
            request_get=request_get,
            url=_GITHUB_URL,
            params={"per_page": 100, "page": page},
        )
        if not isinstance(payload, list):
            raise UpdateCheckError
        if not payload:
            return releases
        for release in payload:
            if not isinstance(release, Mapping):
                raise UpdateCheckError
            releases.append(release)
        page += 1


def _select_release_notes(
    *,
    releases: Sequence[Mapping[str, object]],
    installed: Version,
    latest: Version,
) -> list[tuple[Version, str]]:
    selected: list[tuple[Version, str]] = []
    for release in releases:
        if bool(release.get("draft")) or bool(release.get("prerelease")):
            continue
        tag_name = release.get("tag_name")
        body = release.get("body")
        if not isinstance(tag_name, str) or not isinstance(body, str):
            raise UpdateCheckError
        try:
            version = Version(tag_name.removeprefix("v"))
        except InvalidVersion as exc:
            raise UpdateCheckError from exc
        if installed < version <= latest:
            selected.append((version, body.replace("\r\n", "\n").strip()))
    return sorted(selected, key=lambda item: item[0])


def _format_update_message(
    *, installed: Version, latest: Version, release_notes: Sequence[tuple[Version, str]]
) -> str:
    lines = [
        f"pyaesa update available: {installed} -> {latest}",
        "",
        "Update with:",
        "    python -m pip install --upgrade pyaesa",
    ]
    if release_notes:
        lines.extend(["", "Changes since " + str(installed) + ":", ""])
        for version, body in release_notes:
            lines.extend([f"v{version}", "---------", body, ""])
    else:
        lines.extend(["", "Release notes could not be retrieved from GitHub during this check."])
    lines.extend(["Release history:", _HISTORY_URL])
    return "\n".join(lines)


class UpdateNotifier:
    """Attempt one update check during the lifetime of this interpreter."""

    def __init__(self) -> None:
        self._attempted = False

    def check(
        self,
        *,
        installed_version: str,
        request_get: _RequestGet = requests.get,
        stream: TextIO = sys.stderr,
    ) -> None:
        if self._attempted or _check_disabled():
            return
        self._attempted = True
        try:
            installed = Version(installed_version)
            latest = _fetch_pypi_version(request_get=request_get)
        except (InvalidVersion, UpdateCheckError):
            print("pyaesa update information could not be retrieved from PyPI.", file=stream)
            return
        if installed >= latest:
            return
        try:
            releases = _fetch_github_releases(request_get=request_get)
            notes = _select_release_notes(
                releases=releases,
                installed=installed,
                latest=latest,
            )
        except UpdateCheckError:
            print(
                _format_update_message(installed=installed, latest=latest, release_notes=[]),
                file=stream,
            )
            return
        print(
            _format_update_message(installed=installed, latest=latest, release_notes=notes),
            file=stream,
        )


_NOTIFIER = UpdateNotifier()


def check_for_updates(*, installed_version: str) -> None:
    """Attempt the process-scoped update notification."""
    _NOTIFIER.check(installed_version=installed_version)
