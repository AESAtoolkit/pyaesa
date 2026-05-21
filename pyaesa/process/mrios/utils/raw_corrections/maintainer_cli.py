"""Maintainer CLI logic for EXIOBASE raw corrected values generation."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from .exio_3102_corrected_values import write_corrected_values_outputs
from .runtime import SUPPORTED_SOURCES

DEFAULT_WORKSPACE_ROOT = Path(
    r"C:\Users\Erwan\Documents\UNCASExt_demo"
)  # To update for personal use


def parse_build_corrected_values_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse maintainer arguments for raw corrected values generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Build EXIOBASE 3.10.2 raw corrected values tables from a "
            "workspace root. Edit DEFAULT_WORKSPACE_ROOT at the top of this "
            "module if needed."
        )
    )
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help="Path to the workspace root containing the pyaesa data folders.",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=list(SUPPORTED_SOURCES),
        help="Optional source filter. Defaults to both exiobase_3102_ixi and exiobase_3102_pxp.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory override for the generated corrected values files.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    emit: Callable[[str], None] | None = None,
) -> int:
    """Run the maintainer raw corrected values build."""
    args = parse_build_corrected_values_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    out_dir = None if not str(args.out_dir).strip() else Path(str(args.out_dir)).resolve()
    sources = list(args.source or SUPPORTED_SOURCES)
    log = print if emit is None else emit
    log(f"Workspace root: {workspace_root}")
    for source in sources:
        log(f"Building raw corrected values for {source}...")
        outputs = write_corrected_values_outputs(
            workspace_root=workspace_root,
            source=source,
            out_dir=out_dir,
            progress=lambda message: log(f"  {message}"),
        )
        log(f"  corrected_values: {outputs.corrected_values_path}")
    return 0
