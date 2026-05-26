"""aCC uncertainty source methods and README writing."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.acc.uncertainty.runtime.models import ACCAsoccInput, ACCDynamicCCInput
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import public_run_artifact_readme_lines
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def build_acc_source_methods(
    *,
    asocc_input: ACCAsoccInput,
    dynamic_cc_input: ACCDynamicCCInput | None,
) -> pd.DataFrame:
    """Return upstream source method rows plus the aCC formula row."""
    frames = [_asocc_source_methods(asocc_input=asocc_input)]
    cc_manifest = None if dynamic_cc_input is None else dynamic_cc_input.manifest
    if cc_manifest is not None:
        frames.append(_upstream_source_methods(manifest=cc_manifest))
    frames.append(
        pd.DataFrame(
            [
                {
                    "source_component": "acc",
                    "source_name": "acc_formula",
                    "formula": "aCC = aSoCC * CC",
                    "notes": "aCC uncertainty combines upstream aSoCC shares and CC values.",
                }
            ]
        )
    )
    return pd.concat(frames, ignore_index=True, sort=False)


def _asocc_source_methods(*, asocc_input: ACCAsoccInput) -> pd.DataFrame:
    if asocc_input.manifest is not None:
        return _upstream_source_methods(manifest=cast(UncertaintyManifest, asocc_input.manifest))
    return pd.DataFrame(
        [
            {
                "source_component": "asocc",
                "source_name": "asocc::deterministic_asocc",
                "formula": "aSoCC values are fixed deterministic prerequisite rows.",
            }
        ]
    )


def write_acc_source_methods(*, path: Path, rows: pd.DataFrame) -> None:
    """Write aCC uncertainty source method rows."""
    ensure_file_parent(path)
    rows.to_csv(path, index=False)


def write_acc_results_readme(
    *,
    path: Path,
    active_sources: tuple[str, ...],
    run_layout: str,
) -> None:
    """Write the public aCC uncertainty results README."""
    ensure_file_parent(path)
    run_description = (
        "compact run by public row numeric matrix"
        if run_layout == "compact_run_matrix"
        else "sparse run row table with run_index, public_row_id, and acc"
    )
    lines = [
        "aCC uncertainty results",
        "",
        "Files",
        "- public_row_identity: public aCC row identity table.",
        *public_run_artifact_readme_lines(run_name="acc_runs"),
        f"  Layout: {run_description}.",
        "- summary_stats_runs: exact summary statistics for acc_runs.",
        "- source_methods.csv: upstream and aCC formula source method metadata.",
        "  Rows prefixed with asocc:: and ar6_cc:: keep the upstream source",
        "  method details, including LCIA CoV mapping, inter-MRIO alternate",
        "  source labels, and dynamic AR6 trajectory sampling when active.",
        "- scope_manifest.json: run identity, inputs, outputs, convergence status,",
        "  and canonical public table schemas for this result scope.",
        "",
        "Method",
        "aCC uncertainty evaluates aCC = aSoCC * CC. Static carrying",
        "capacity values come from the selected carrying capacity table.",
        "Dynamic AR6 carrying capacity values use the matching",
        "uncertainty_ar6_cc run values.",
        "",
        "Active source dimensions",
        *(f"- {source}" for source in active_sources),
        "",
    ]
    path.write_text(join_user_text_lines(lines), encoding="utf-8")


def _upstream_source_methods(*, manifest: UncertaintyManifest) -> pd.DataFrame:
    artifacts = cast(dict[str, Any], manifest.artifacts)
    rows = pd.read_csv(Path(str(artifacts["source_methods"]))).copy()
    source_prefix = f"{manifest.family}::"
    rows["source_name"] = rows["source_name"].map(
        lambda value: _qualified_source_name(value=value, prefix=source_prefix)
    )
    return rows


def _qualified_source_name(*, value: object, prefix: str) -> str:
    text = str(value).strip()
    return text if text.startswith(prefix) else f"{prefix}{text}"
