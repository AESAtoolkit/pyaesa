"""Metadata ownership for MRIO downloaders.

The module allows callers to read the JSON payload and write coverage
after downloads.
"""

from datetime import datetime
import json
from typing import Any, Dict, Iterable, Optional, cast

from pyaesa.download.mrios.utils.paths import _get_metadata_path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


def _read_meta(source_key: str) -> Optional[Dict[str, Any]]:
    """Return metadata JSON for ``source_key`` or ``None`` when the file is missing."""
    path = _get_metadata_path(source_key)
    if not path.exists():
        return None
    return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_meta(source_key: str, years: Iterable[int]) -> None:
    """Write metadata for ``source_key`` covering the provided ``years``."""
    path = ensure_file_parent(_get_metadata_path(source_key))
    years_list = sorted({int(y) for y in years})
    payload: Dict[str, Any] = {
        "source": source_key,
        "years": years_list,
        "timestamp": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
