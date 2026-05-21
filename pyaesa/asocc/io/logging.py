"""File logger ownership for allocation methods."""

import logging
from pathlib import Path

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

_ALLOCATE_LOGGER_NAMES: set[str] = set()


def get_logger(log_path: Path) -> logging.Logger:
    """Return a configured file logger for one allocation run."""
    logger_name = f"deterministic_asocc:{log_path}"
    logger = logging.getLogger(logger_name)
    _ALLOCATE_LOGGER_NAMES.add(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    resolved = ensure_file_parent(log_path)
    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def close_loggers_for_scope(scope_root: Path) -> None:
    """Close allocation file handlers that live under one filesystem scope."""
    scope_root_resolved = scope_root.resolve()
    for logger_name in list(_ALLOCATE_LOGGER_NAMES):
        logger_obj = logging.getLogger(logger_name)
        for handler in list(logger_obj.handlers):
            base_filename = getattr(handler, "baseFilename", None)
            if not isinstance(base_filename, str):
                continue
            handler_path = Path(base_filename).resolve()
            if not handler_path.is_relative_to(scope_root_resolved):
                continue
            handler.close()
            logger_obj.removeHandler(handler)
        if not logger_obj.handlers:
            _ALLOCATE_LOGGER_NAMES.discard(logger_name)
