"""File identity payloads for manifest compatibility keys."""

import hashlib
from pathlib import Path


def file_identity_payload(*, path: Path) -> dict[str, object]:
    """Return path, size, and SHA256 identity for one input file."""
    file_path = Path(path)
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": file_sha256(path=file_path),
    }


def file_sha256(*, path: Path) -> str:
    """Return the SHA256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
