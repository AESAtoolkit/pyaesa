"""Internal archive validation ownership for MRIO downloads."""

from pathlib import Path
import zipfile


def _assert_valid_zip(archive_path: Path, *, artifact_label: str) -> None:
    """Raise when ``archive_path`` is not a readable ZIP archive."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_member = archive.testzip()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"{artifact_label} at {archive_path} is not a valid ZIP archive."
        ) from exc
    if bad_member is not None:
        raise RuntimeError(
            f"{artifact_label} at {archive_path} is corrupted inside member {bad_member!r}."
        )
