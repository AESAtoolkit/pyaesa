"""Artifact writing for aSoCC inter-method probability trees."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.asocc.inter_method_tools.tree import (
    InterMethodCandidate,
    inter_method_tree_probabilities,
    write_inter_method_tree_csv,
)
from pyaesa.asocc.inter_method_tools.tree_figure import render_inter_method_tree
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE
from pyaesa.shared.figures.request_validation import normalize_figure_format
from pyaesa.shared.runtime.io.filesystem import atomic_write_text
from pyaesa.shared.runtime.text import join_user_text_lines

INTER_METHOD_TREE_GUIDE_NAME = "README_inter_method_weights.txt"


@dataclass(frozen=True)
class InterMethodTreeArtifacts:
    """Written inter-method probability tree artifacts."""

    tree_csv_path: Path
    guide_path: Path
    figure_paths: tuple[Path, ...]
    candidates: tuple[str, ...]
    probabilities: tuple[float, ...]
    summary_lines: tuple[str, ...]


def write_inter_method_tree_artifacts(
    *,
    tree_csv_path: Path,
    figure_base_path: Path,
    frame: pd.DataFrame,
    candidates: tuple[InterMethodCandidate, ...],
    figure_format: dict[str, Any] | None,
) -> InterMethodTreeArtifacts:
    """Write one exact tree CSV and matching tree figure."""
    figure = normalize_figure_format(figure_format)
    write_inter_method_tree_csv(path=tree_csv_path, frame=frame)
    guide_path = write_inter_method_tree_guide(tree_csv_path=tree_csv_path)
    figure_paths = tuple(
        render_inter_method_tree(
            frame=frame,
            figure_base_path=figure_base_path,
            output_format=str(figure["format"]),
            dpi=int(figure["dpi"]),
        )
    )
    labels = tuple(candidate.candidate_label for candidate in candidates)
    probs = tuple(
        float(value)
        for value in inter_method_tree_probabilities(frame=frame, candidates=candidates)
    )
    return InterMethodTreeArtifacts(
        tree_csv_path=tree_csv_path,
        guide_path=guide_path,
        figure_paths=figure_paths,
        candidates=labels,
        probabilities=probs,
        summary_lines=(
            f"tree_csv={tree_csv_path}",
            f"guide={guide_path}",
            f"figure_paths={[str(path) for path in figure_paths]}",
            f"candidate_count={len(labels)}",
        ),
    )


def inter_method_tree_guide_path(*, tree_csv_path: Path) -> Path:
    """Return the guide path placed beside one inter-method tree CSV."""
    return Path(tree_csv_path).with_name(INTER_METHOD_TREE_GUIDE_NAME)


def write_inter_method_tree_guide(*, tree_csv_path: Path) -> Path:
    """Write the user guide for editing one inter-method tree CSV."""
    guide_path = inter_method_tree_guide_path(tree_csv_path=tree_csv_path)
    return atomic_write_text(
        guide_path,
        text=_inter_method_tree_guide_text(tree_csv_path=tree_csv_path),
    )


def _inter_method_tree_guide_text(*, tree_csv_path: Path) -> str:
    csv_name = Path(tree_csv_path).name
    return join_user_text_lines(
        f"""aSoCC inter-method weight tree guide

This folder contains an editable probability tree CSV:

{csv_name}

Purpose
=======

The CSV defines how inter-method uncertainty samples allocation method leaves.
The tree is used by uncertainty_asocc(...) when {INTER_METHOD_SOURCE} is
active. The default export gives equal weight to the branch hierarchy, not flat
equal weight to every final method leaf. It follows the principle of equal
weight per sharing principle and then within each sharing principle equal
weight per enacting metric (Puig-Samper et al., 2025; de Bantel et al., 2026).

How to edit weights
===================

Edit only the edge_weight column. Keep parent_id, node_id, label, node_type,
level, and candidate_label unchanged.

For every parent_id, the edge_weight values of its direct children must sum to
1. Values must be between 0 and 1. A final method probability is the product of
edge_weight values from root to the terminal row carrying candidate_label.

Use preview_asocc_weight_tree(...) after editing a custom CSV. For a custom
version named "custom_v1", place the edited file in this folder at:

weights__custom_v1.csv

Then run:

preview_asocc_weight_tree(
    base_asocc_args=...,
    version_name="custom_v1",
    external_method=...,
)

Use the same version in uncertainty_asocc(...):

uncertainty_config={{
    "{INTER_METHOD_SOURCE}": {{
        "mode": "custom",
        "version_name": "custom_v1",
    }},
}}

CSV columns
===========

parent_id:
    Parent node identifier. root is the artificial start node.

node_id:
    Stable identifier for this node. Do not edit it.

label:
    Display label shown in the figure. Do not edit it.

node_type:
    Internal node kind used by the renderer. Do not edit it.

edge_weight:
    Conditional probability from parent_id to node_id. This is the editable
    probability field.

level:
    Tree depth used by the renderer. Do not edit it.

candidate_label:
    Final method leaf sampled by inter-method uncertainty. Empty rows are
    internal branch nodes.

Tree classification
===================

Method labels are parsed as:

<sharing principle>(<enacting metric>)
<sharing principle>-<subprinciple>(<enacting metric>)

Examples:

EG(Pop)
    sharing principle EG, enacting metric Pop.

PR-HR(Ecap,cum)
    sharing principle PR, subprinciple HR, enacting metric Ecap,cum.

PR-HR(Ecap,cum^{{PBA}})
    sharing principle PR, subprinciple HR, enacting metric Ecap,cum_PBA.

AR(E^{{CBA_FD}})
    sharing principle AR, enacting metric E_CBA_FD.

For L2 functional units, a combined method such as:

PR-HR(Ecap,cum^{{PBA}})::UT(GVAa)

is represented as an L1 branch PR -> HR -> Ecap,cum_PBA, then an L2 branch
UT -> GVAa. The terminal candidate_label uses the package public label:

PR-HR(Ecap,cum^{{PBA}})_UT(GVAa)

L2 path buckets
===============

When one sharing principle contains both multi step candidates and one step
candidates, the tree inserts two branch nodes:

m_s:
    multi step allocation path.

o_s:
    one step allocation path.

The default export gives m_s and o_s equal conditional weight. You can change
those two edge_weight values as long as they still sum to 1 under the same
parent.

Optional external leaves
========================

External aSoCC methods enter this CSV only when you pass the same
external_method selector to write_asocc_weight_template(...),
preview_asocc_weight_tree(...), and uncertainty_asocc(...).

L1 external example:

external_method={{"l1_methods": ["CO-HR(S,cum)"]}}

L2 one step external example:

external_method={{"one_step_methods": ["CO(S)"]}}

L2 pair external example:

external_method={{
    "l1_l2_pairs": ["PR-HR(Ecap,cum^{{PBA}})::UT(GVAa)"],
}}

External labels are classified by the same parser as package labels. Prose
labels are not valid because the tree must know the sharing principle,
optional subprinciple, and enacting metric.
""".splitlines(),
        trailing_newline=True,
    )
