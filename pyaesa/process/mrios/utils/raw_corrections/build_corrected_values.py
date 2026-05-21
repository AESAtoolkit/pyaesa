"""Executable maintainer entrypoint for EXIOBASE raw corrected values tables.

Run with:

``python -m pyaesa.process.mrios.utils.raw_corrections.build_corrected_values``

The reusable argument parsing and generation logic lives in
``maintainer_cli.py`` so maintainer code can import that logic without loading
the executable module path.
"""

from .maintainer_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
