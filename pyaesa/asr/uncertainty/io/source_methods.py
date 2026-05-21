"""ASR uncertainty source method and README writers."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan
from pyaesa.asr.uncertainty.sources.source_keys import acc_source_name, io_lca_source_name
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.io.tables import public_run_artifact_readme_lines


def build_asr_source_methods(*, plan: ASRUncertaintyPlan) -> pd.DataFrame:
    """Return ASR source method rows from aCC, LCA, and formula ownership."""
    rows = []
    rows.append(_acc_source_methods(plan=plan))
    if not plan.lca_input.source_method_rows.empty:
        rows.append(_lca_source_methods(plan.lca_input.source_method_rows))
    rows.append(
        pd.DataFrame.from_records(
            [
                {
                    "source_component": "asr",
                    "source_name": "asr_formula",
                    "formula": "ASR = LCA / aCC",
                    "notes": (
                        "ASR combines upstream aCC denominator runs and LCA numerator "
                        "runs by matched public row identity."
                    ),
                }
            ]
        )
    )
    columns = list(dict.fromkeys(column for frame in rows for column in frame.columns))
    return pd.concat([frame.reindex(columns=columns) for frame in rows], ignore_index=True)


def _acc_source_methods(*, plan: ASRUncertaintyPlan) -> pd.DataFrame:
    acc_artifacts = cast(dict[str, Any], plan.acc_manifest.artifacts)
    rows = pd.read_csv(acc_artifacts["source_methods"]).copy()
    rows["source_component"] = "acc"
    rows["source_name"] = rows["source_name"].map(acc_source_name)
    return rows


def _lca_source_methods(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    io_mask = out["source_component"].astype(str).eq("io_lca")
    out.loc[io_mask, "source_name"] = out.loc[io_mask, "source_name"].map(io_lca_source_name)
    return out


def write_asr_source_methods(*, path: Path, rows: pd.DataFrame) -> None:
    """Write the ASR source method log."""
    ordered = rows.sort_values(
        [column for column in ("source_component", "source_name") if column in rows.columns],
        kind="mergesort",
    ).reset_index(drop=True)
    write_via_atomic_temp(
        ensure_file_parent(path),
        writer=lambda tmp_path: ordered.to_csv(tmp_path, index=False),
    )


def write_asr_results_readme(
    *,
    path: Path,
    active_sources: tuple[str, ...],
    run_layout: str,
    include_cumulative: bool,
) -> None:
    """Write the public ASR Monte Carlo result guide."""
    artifact_lines = [
        "- public_row_identity: public ASR rows, one row per run matrix column.",
        *public_run_artifact_readme_lines(run_name="asr_runs"),
        "  Layout: ASR values by run and public row.",
        "- summary_stats_runs: exact yearly ASR and frequency of no transgression",
        "  outputs computed from all runs. ASR rows carry summary statistics;",
        "  frequency rows carry the fNT fraction.",
    ]
    if include_cumulative:
        artifact_lines.extend(
            [
                "- cumulative_row_identity: yearless full study period ASR identities.",
                *public_run_artifact_readme_lines(run_name="cumulative_asr_runs"),
                "  Layout: cumulative ASR values by run and period identity.",
                "- cumulative_summary_stats_runs: cumulative ASR and frequency of",
                "  no-transgression outputs computed from all cumulative runs. Cumulative",
                "  ASR rows carry summary statistics; frequency rows carry the fNT fraction.",
            ]
        )
    artifact_lines.extend(
        [
            "- source_methods.csv: upstream aCC, LCA, and ASR formula source records.",
            "  Upstream rows keep their qualified source names, such as acc::,",
            "  asocc::, ar6_cc::, and io_lca::. External LCA rows identify the",
            "  user supplied version label when external LCA uncertainty is active,",
            "  but external LCA provenance is owned by the user supplied files.",
            "  Inspect upstream package rows for LCIA CoV mapping, inter-MRIO",
            "  alternate source labels, or dynamic AR6 trajectory sampling.",
            "- scope_manifest.json: request, prerequisite, output, reuse metadata,",
            "  and canonical public table schemas for this result scope.",
        ]
    )
    lines = [
        "ASR Uncertainty Results",
        "",
        "This run evaluates absolute sustainability ratio uncertainty as:",
        "ASR = LCA / aCC",
        "",
        "Artifacts",
        *artifact_lines,
        "",
        "Run Layout",
        f"- {run_layout}",
        "",
        "Active Sources",
        *[f"- {source}" for source in active_sources],
        "",
        "Interpretation",
        "LCA values are converted to the matched aCC impact unit when a supported",
        "unit conversion exists. ASR values are undefined when the matched aCC",
        "denominator is zero.",
        "",
    ]

    def _write_text(tmp_path: Path) -> None:
        tmp_path.write_text(join_user_text_lines(lines), encoding="utf-8")

    write_via_atomic_temp(ensure_file_parent(path), writer=_write_text)
