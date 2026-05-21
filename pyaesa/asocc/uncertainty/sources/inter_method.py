"""Inter-method uncertainty as compact method leaf selection."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.asocc.inter_method_tools.tree import (
    DEFAULT_INTER_METHOD_TREE_VERSION,
    InterMethodCandidate,
    candidates_from_rows,
    build_inter_method_tree_frame,
    inter_method_tree_probabilities,
    inter_method_tree_version_name,
    inter_method_tree_path,
    load_valid_inter_method_tree_frame,
)
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import LoadedAsoccFinalRows
from pyaesa.asocc.uncertainty.inputs.external_rows import ExternalAsoccRowsPlan
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE

INTER_METHOD_SELECTION_RANDOM_STREAM = "asocc.inter_method.selection"
_METHOD_COLUMNS = ("l1_l2_method", "l1_method", "l2_method")


@dataclass(frozen=True)
class InterMethodPlan:
    """Resolved method leaf probability plan for inter-method uncertainty."""

    candidates: tuple[InterMethodCandidate, ...]
    candidate_labels: tuple[str, ...]
    selection_probabilities: np.ndarray
    tree_frame: pd.DataFrame
    source_method_row: SourceMethodRow


def build_inter_method_plan(
    *,
    loaded: LoadedAsoccFinalRows,
    parameters: dict[str, Any],
    external_plan: ExternalAsoccRowsPlan | None = None,
) -> InterMethodPlan:
    """Build the method leaf selection plan from public method labels."""
    candidates = _candidate_inventory(
        rows=loaded.rows,
        external_plan=external_plan or ExternalAsoccRowsPlan(),
    )
    labels = tuple(candidate.candidate_label for candidate in candidates)
    tree_frame, probabilities = _candidate_tree_frame_and_probabilities(
        candidates=candidates,
        loaded=loaded,
        parameters=parameters,
    )
    return InterMethodPlan(
        candidates=candidates,
        candidate_labels=labels,
        selection_probabilities=probabilities,
        tree_frame=tree_frame,
        source_method_row=inter_method_source_method_row(loaded=loaded, labels=labels),
    )


def inter_method_uncertainty_has_targets(
    *,
    loaded: LoadedAsoccFinalRows,
    external_plan: ExternalAsoccRowsPlan | None = None,
) -> bool:
    """Return whether selected rows expose at least two method candidates."""
    return (
        len(
            _candidate_inventory(
                rows=loaded.rows,
                external_plan=external_plan or ExternalAsoccRowsPlan(),
            )
        )
        >= 2
    )


def inter_method_source_method_row(
    *,
    loaded: LoadedAsoccFinalRows,
    labels: tuple[str, ...],
) -> SourceMethodRow:
    """Return the compact scientific log row for inter-method uncertainty."""
    return SourceMethodRow(
        source_component="asocc",
        source_name=INTER_METHOD_SOURCE,
        scope=str(loaded.base_asocc_args["fu_code"]),
        applied_bucket=loaded.final_bucket,
        year_min=min(loaded.requested_years),
        year_max=max(loaded.requested_years),
        distribution="discrete probability over selected deterministic methods",
        shared_random_variable="run_index",
        formula="sampled row = active-source aSoCC value for selected method leaf",
        notes=f"Candidate method leaves: {';'.join(labels)}",
    )


def _candidate_tree_frame_and_probabilities(
    *,
    candidates: tuple[InterMethodCandidate, ...],
    loaded: LoadedAsoccFinalRows,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    version_name = inter_method_tree_version_name(parameters=parameters)
    if version_name == DEFAULT_INTER_METHOD_TREE_VERSION:
        frame = build_inter_method_tree_frame(candidates=candidates)
        return frame, np.asarray(
            inter_method_tree_probabilities(frame=frame, candidates=candidates),
            dtype=np.float64,
        )
    return load_valid_inter_method_tree_frame(
        candidates=candidates,
        custom_path=inter_method_tree_path(
            proj_base=loaded.path_scope.proj_base,
            version_name=version_name,
        ),
    )


def inter_method_row_labels(*, rows: pd.DataFrame) -> np.ndarray:
    """Return selected method leaf labels for public aSoCC rows."""
    return _row_labels(rows=rows)


def sample_inter_method_labels(
    *,
    plan: InterMethodPlan,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Sample one method leaf label for each run in a batch."""
    cumulative = np.cumsum(plan.selection_probabilities)
    cumulative[-1] = 1.0
    uniform = (
        np.asarray(unit_values, dtype=np.float64)
        if unit_values is not None
        else uniform_by_run_index(
            stream_name=INTER_METHOD_SELECTION_RANDOM_STREAM,
            run_indices=batch.run_indices(),
        )
    )
    indices = np.searchsorted(
        cumulative,
        uniform,
        side="right",
    )
    return np.array(plan.candidate_labels, dtype=object)[indices]


def _candidate_inventory(
    *,
    rows: pd.DataFrame,
    external_plan: ExternalAsoccRowsPlan,
) -> tuple[InterMethodCandidate, ...]:
    native_candidates = candidates_from_rows(rows=rows)
    # External Monte Carlo rows do not live in loaded.rows. Keep the declared
    # level and method metadata from the external file selection so L2 one step
    # external methods are not reclassified from their text label alone.
    external_candidates = tuple(
        InterMethodCandidate(
            candidate_label=source.selection.asocc_method_label,
            level=source.selection.level,
            l1_method=source.selection.l1_method,
            l2_method=source.selection.l2_method,
        )
        for source in external_plan.monte_carlo_sources
    )
    by_label = {
        candidate.candidate_label: candidate
        for candidate in (*native_candidates, *external_candidates)
    }
    return tuple(by_label[label] for label in sorted(by_label))


def _row_labels(*, rows: pd.DataFrame) -> np.ndarray:
    labels = (
        pd.Series(rows.loc[:, "l2_method"], copy=False)
        .astype(str)
        .to_numpy(
            dtype=object,
            copy=True,
        )
    )
    for column in reversed(_METHOD_COLUMNS[:-1]):
        if column in rows.columns:
            values = pd.Series(rows.loc[:, column], copy=False)
            mask = values.notna().to_numpy(dtype=bool)
            labels[mask] = values.loc[mask].astype(str).to_numpy(dtype=object)
    return labels
