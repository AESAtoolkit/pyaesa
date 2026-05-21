"""Logger ownership for MRIO download and parsing calls."""

from contextlib import contextmanager
import logging
import warnings


@contextmanager
def suppress_pymrio_logging():
    """Temporarily suppress noisy third party logger output and known external warnings."""
    old = logging.root.manager.disable
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            module=r"pymrio\.tools\.iomath",
        )
        try:
            logging.disable(logging.CRITICAL)
            yield
        finally:
            logging.disable(old)
