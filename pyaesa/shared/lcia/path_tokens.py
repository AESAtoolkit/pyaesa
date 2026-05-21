"""LCIA method token inference from deterministic artifact paths."""

from pathlib import Path
import re

from pyaesa.shared.lcia.availability import discover_static_cc_methods

_LCIA_SUFFIX_RE_TEMPLATE = r"(?:_|__){lcia_method_token}(?:(?:_|__)[^_]+)?$"


def known_lcia_methods() -> tuple[str, ...]:
    """Return known bundled LCIA method tokens sorted for suffix matching."""
    return tuple(sorted(discover_static_cc_methods(), key=len, reverse=True))


def infer_lcia_method_from_path(path: Path) -> str | None:
    """Infer the LCIA method token encoded in one persisted file stem."""
    stem = str(path.stem).strip()
    path_text = str(path).strip()
    path_parts = {str(part).strip() for part in path.parts if str(part).strip()}
    matches = [
        lcia_method_token
        for lcia_method_token in known_lcia_methods()
        if stem.endswith(f"_{lcia_method_token}")
        or stem.endswith(f"__{lcia_method_token}")
        or f"__{lcia_method_token}__" in stem
        or f"_{lcia_method_token}__" in stem
        or str(lcia_method_token).strip() in path_parts
        or f"\\{str(lcia_method_token).strip()}\\" in path_text
        or f"/{str(lcia_method_token).strip()}/" in path_text
        or re.search(
            _LCIA_SUFFIX_RE_TEMPLATE.format(lcia_method_token=re.escape(lcia_method_token)),
            stem,
        )
    ]
    if not matches:
        return None
    return matches[0]
