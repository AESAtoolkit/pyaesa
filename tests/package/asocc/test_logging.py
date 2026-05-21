import logging
from pathlib import Path

from pyaesa.asocc.io import logging as logging_mod


def test_close_loggers_for_scope_skips_non_file_handlers_and_closes_in_scope_files(
    tmp_path: Path,
) -> None:
    scope_root = tmp_path / "scope"
    log_path = scope_root / "summary.log"
    logger = logging_mod.get_logger(log_path)
    same_logger = logging_mod.get_logger(log_path)
    outside_logger = logging_mod.get_logger(tmp_path / "outside" / "keep.log")
    null_handler = logging.NullHandler()
    logger.addHandler(null_handler)

    assert same_logger is logger
    assert sum(isinstance(handler, logging.FileHandler) for handler in logger.handlers) == 1
    assert any(isinstance(handler, logging.FileHandler) for handler in outside_logger.handlers)

    logging_mod.close_loggers_for_scope(scope_root)

    assert null_handler in logger.handlers
    assert not any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None)
        for handler in logger.handlers
    )
    assert any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None)
        for handler in outside_logger.handlers
    )

    logger.removeHandler(null_handler)
    logging_mod.close_loggers_for_scope(scope_root)
    logging_mod.close_loggers_for_scope(tmp_path / "outside")
