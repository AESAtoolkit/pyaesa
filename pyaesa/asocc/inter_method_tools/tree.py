"""Canonical aSoCC inter-method probability tree ownership."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.asocc.runtime.methods.labels import parse_raw_asocc_method_label
from pyaesa.asocc.runtime.paths.family_roots import _get_asocc_root
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.runtime.request.scope import AsoccScope, build_asocc_scope
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE
from pyaesa.external_inputs.asocc.schema.contracts import (
    iter_external_method_selections,
    validate_external_method_collisions,
)
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.tabular.scalars import is_display_missing

DEFAULT_INTER_METHOD_TREE_VERSION = "equal_weight_default"
DEFAULT_INTER_METHOD_TREE_CSV_NAME = "equal_weights.csv"
INTER_METHOD_TREE_COLUMNS = (
    "parent_id",
    "node_id",
    "label",
    "node_type",
    "edge_weight",
    "level",
    "candidate_label",
)
EDGE_WEIGHT_TOLERANCE = 1e-9
INTER_METHOD_TREE_MODES = ("equal_weight", "custom")


@dataclass(frozen=True)
class InterMethodCandidate:
    """One final method leaf represented in the inter-method tree."""

    candidate_label: str
    level: str
    l1_method: str | None
    l2_method: str | None


@dataclass(frozen=True)
class InterMethodTreeRequest:
    """Resolved method candidates and output root for one public tree request."""

    proj_base: Path
    candidates: tuple[InterMethodCandidate, ...]


@dataclass(frozen=True)
class _PlannedMethod:
    """Parsed method label used while constructing the probability tree."""

    method_name: str
    sharing_principle: str
    subprinciple: str | None
    enacting_metric: str
    candidate_label: str | None


def inter_method_tree_version_name(*, parameters: dict[str, Any] | None) -> str:
    """Return the validated probability tree version for inter-method parameters."""
    if parameters is None:
        return DEFAULT_INTER_METHOD_TREE_VERSION
    params = dict(parameters)
    mode = str(params.pop("mode", "equal_weight")).strip() or "equal_weight"
    if mode not in INTER_METHOD_TREE_MODES:
        raise ValueError(f"{INTER_METHOD_SOURCE}.mode must be 'equal_weight' or 'custom'.")
    if mode == "equal_weight":
        if params:
            raise ValueError(f"Unsupported {INTER_METHOD_SOURCE} parameter(s): {sorted(params)}.")
        return DEFAULT_INTER_METHOD_TREE_VERSION
    version_name = _validate_custom_version_name(params.pop("version_name", None))
    if params:
        raise ValueError(f"Unsupported {INTER_METHOD_SOURCE} parameter(s): {sorted(params)}.")
    return version_name


def inter_method_tree_path(*, proj_base: Path, version_name: str) -> Path:
    """Return the editable inter-method tree CSV path for one version."""
    root = _get_asocc_root(proj_base=proj_base) / "preview_inter_method_weights"
    return root / inter_method_tree_csv_name(version_name=version_name)


def inter_method_preview_figure_base(*, proj_base: Path, version_name: str) -> Path:
    """Return the preview figure base path for one inter-method tree version."""
    root = _get_asocc_root(proj_base=proj_base) / "preview_inter_method_weights"
    return root / inter_method_tree_figure_stem(version_name=version_name)


def inter_method_tree_csv_name(*, version_name: str) -> str:
    """Return the canonical inter-method tree CSV file name for one version."""
    version = _validate_tree_version_name(version_name)
    if version == DEFAULT_INTER_METHOD_TREE_VERSION:
        return DEFAULT_INTER_METHOD_TREE_CSV_NAME
    return f"weights__{version}.csv"


def inter_method_tree_figure_stem(*, version_name: str) -> str:
    """Return the canonical inter-method tree figure stem for one version."""
    version = _validate_tree_version_name(version_name)
    if version == DEFAULT_INTER_METHOD_TREE_VERSION:
        return "probability_tree__equal_weights"
    return f"probability_tree__{version}"


def candidates_from_scope(
    *,
    base_asocc_args: dict[str, Any],
    external_method: dict[str, Any] | None = None,
) -> tuple[InterMethodCandidate, ...]:
    """Return inter-method candidates from normalized deterministic selectors."""
    return plan_inter_method_tree_request(
        base_asocc_args=base_asocc_args,
        external_method=external_method,
    ).candidates


def plan_inter_method_tree_request(
    *,
    base_asocc_args: dict[str, Any],
    external_method: dict[str, Any] | None = None,
) -> InterMethodTreeRequest:
    """Resolve one public inter-method tree request once."""
    normalized = normalize_base_allocate_args(base_asocc_args)
    scope = build_asocc_scope(base_allocate_args=normalized)
    candidates = _native_candidates_from_scope(scope=scope)
    if external_method is not None:
        validate_external_method_collisions(
            native_labels=[candidate.candidate_label for candidate in candidates],
            external_method=external_method,
            fu_code=str(normalized["fu_code"]),
            where="aSoCC inter-method probability tree",
        )
        candidates.extend(
            InterMethodCandidate(
                candidate_label=selection.asocc_method_label,
                level=selection.level,
                l1_method=selection.l1_method,
                l2_method=selection.l2_method,
            )
            for selection in iter_external_method_selections(
                external_method=external_method,
                fu_code=str(normalized["fu_code"]),
            )
        )
    return InterMethodTreeRequest(
        proj_base=scope.resolve_path_scope().proj_base,
        candidates=_ordered_candidates(candidates),
    )


def candidates_from_rows(
    *,
    rows: pd.DataFrame,
) -> tuple[InterMethodCandidate, ...]:
    """Return inter-method candidates represented by final public rows."""
    candidates: list[InterMethodCandidate] = []
    for values in (
        rows.loc[
            :, [column for column in ("l1_l2_method", "l1_method", "l2_method") if column in rows]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        row = values._asdict()
        label = (
            _text_or_none(row.get("l1_l2_method"))
            or _text_or_none(row.get("l1_method"))
            or str(row.get("l2_method"))
        )
        l1_method = _text_or_none(row.get("l1_method"))
        l2_method = _text_or_none(row.get("l2_method"))
        candidates.append(
            _candidate_from_label(
                label=label,
                l1_method=l1_method,
                l2_method=l2_method,
            )
        )
    return _ordered_candidates(candidates)


def build_inter_method_tree_frame(
    *,
    candidates: tuple[InterMethodCandidate, ...],
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return one probability tree frame for final method candidates."""
    frame = pd.DataFrame(
        _default_tree_rows(candidates=candidates),
        columns=list(INTER_METHOD_TREE_COLUMNS),
    )
    if probabilities is not None:
        labels = tuple(candidate.candidate_label for candidate in candidates)
        leaf_probabilities = np.asarray(probabilities, dtype=np.float64)
        frame["edge_weight"] = _edge_weights_from_leaf_probabilities(
            frame=frame,
            leaf_probabilities={
                label: float(probability)
                for label, probability in zip(labels, leaf_probabilities, strict=True)
            },
        )
    return frame


def default_inter_method_tree_probabilities(
    *,
    candidates: tuple[InterMethodCandidate, ...],
) -> np.ndarray:
    """Return equal branch probabilities aligned to final method candidates."""
    frame = build_inter_method_tree_frame(candidates=candidates)
    return inter_method_tree_probabilities(frame=frame, candidates=candidates)


def inter_method_tree_probabilities(
    *,
    frame: pd.DataFrame,
    candidates: tuple[InterMethodCandidate, ...],
) -> np.ndarray:
    """Return leaf probabilities from tree edge weights aligned to candidates."""
    probability_by_label = _leaf_probabilities_from_edges(frame=frame)
    return np.array(
        [probability_by_label[candidate.candidate_label] for candidate in candidates],
        dtype=np.float64,
    )


def load_inter_method_tree_probabilities(
    *,
    candidates: tuple[InterMethodCandidate, ...],
    custom_path: Path,
) -> np.ndarray:
    """Return probabilities aligned to candidate labels from one tree CSV."""
    _frame, probabilities = load_valid_inter_method_tree_frame(
        candidates=candidates,
        custom_path=custom_path,
    )
    return probabilities


def load_valid_inter_method_tree_frame(
    *,
    candidates: tuple[InterMethodCandidate, ...],
    custom_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a validated tree frame and probabilities aligned to candidates."""
    expected = build_inter_method_tree_frame(candidates=candidates)
    if not custom_path.exists():
        raise ValueError(f"Custom inter-method tree CSV not found: {custom_path}.")
    actual = pd.read_csv(custom_path).loc[:, list(INTER_METHOD_TREE_COLUMNS)].copy()
    _validate_tree_topology(expected=expected, actual=actual)
    probabilities = inter_method_tree_probabilities(frame=actual, candidates=candidates)
    return actual, probabilities


def write_inter_method_tree_csv(*, path: Path, frame: pd.DataFrame) -> None:
    """Write one inter-method tree CSV."""
    out = frame.loc[:, list(INTER_METHOD_TREE_COLUMNS)]
    target = ensure_file_parent(path)
    write_via_atomic_temp(target, writer=lambda tmp_path: out.to_csv(tmp_path, index=False))


def _native_candidates_from_scope(*, scope: AsoccScope) -> list[InterMethodCandidate]:
    if str(scope.base_allocate_args["fu_code"]).startswith("L1."):
        return [
            InterMethodCandidate(
                candidate_label=str(method),
                level="level_1",
                l1_method=str(method),
                l2_method=None,
            )
            for method in scope.selected_l1
        ]
    return [
        InterMethodCandidate(
            candidate_label=f"{l1_method}_{l2_method}",
            level="level_2",
            l1_method=str(l1_method),
            l2_method=str(l2_method),
        )
        for l2_method, l1_method in scope.combined
    ] + [
        InterMethodCandidate(
            candidate_label=str(method),
            level="level_2",
            l1_method=None,
            l2_method=str(method),
        )
        for method in scope.selected_l2_one_step
    ]


def _ordered_candidates(candidates: list[InterMethodCandidate]) -> tuple[InterMethodCandidate, ...]:
    by_label: dict[str, InterMethodCandidate] = {}
    for candidate in candidates:
        by_label[candidate.candidate_label] = candidate
    return tuple(by_label[label] for label in sorted(by_label))


def _validate_tree_version_name(value: object) -> str:
    text = str(value).strip()
    if text == DEFAULT_INTER_METHOD_TREE_VERSION:
        return text
    return _validate_custom_version_name(text)


def _validate_custom_version_name(value: object) -> str:
    if value is None:
        raise ValueError(f"{INTER_METHOD_SOURCE}.version_name is required when mode='custom'.")
    text = str(value).strip()
    if not text:
        raise ValueError("Inter-method custom version_name must be non empty.")
    if text == DEFAULT_INTER_METHOD_TREE_VERSION:
        raise ValueError(
            "Inter-method custom version_name must not be the equal weight default token."
        )
    if any(not (char.isalnum() or char == "_") for char in text):
        raise ValueError(
            "Inter-method custom version_name may contain only letters, digits, and underscores."
        )
    return text


def _candidate_from_label(
    *,
    label: str,
    l1_method: str | None,
    l2_method: str | None,
) -> InterMethodCandidate:
    resolved_l1 = l1_method
    resolved_l2 = l2_method
    if resolved_l2 is None and ")_" in label:
        split_at = label.index(")_") + 1
        resolved_l1 = label[:split_at]
        resolved_l2 = label[split_at + 1 :]
    if resolved_l2 is None and resolved_l1 is None:
        resolved_l1 = label
    return InterMethodCandidate(
        candidate_label=label,
        level="level_1" if resolved_l2 is None else "level_2",
        l1_method=resolved_l1,
        l2_method=resolved_l2,
    )


def _default_tree_rows(
    *,
    candidates: tuple[InterMethodCandidate, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        *,
        parent_id: str,
        node_id: str,
        label: str,
        node_type: str,
        edge_weight: float,
        level: int,
        candidate_label: str | None = None,
    ) -> None:
        rows.append(
            {
                "parent_id": parent_id,
                "node_id": node_id,
                "label": label,
                "node_type": node_type,
                "edge_weight": float(edge_weight),
                "level": int(level),
                "candidate_label": candidate_label,
            }
        )

    is_l2_fu = any(candidate.level == "level_2" for candidate in candidates)
    first_stage = _first_stage_methods(candidates=candidates)
    standalone = _standalone_methods(candidates=candidates)
    second_by_l1 = _second_step_methods_by_l1(candidates=candidates)
    present_principles = _ordered_unique(
        [item.sharing_principle for item in (*first_stage, *standalone)]
    )
    root_weight = 1.0 / len(present_principles)
    for principle in present_principles:
        add(
            parent_id="root",
            node_id=principle,
            label=principle,
            node_type="principle",
            edge_weight=root_weight,
            level=1,
        )

    first_by_principle: dict[str, list[_PlannedMethod]] = {p: [] for p in present_principles}
    for item in first_stage:
        first_by_principle[item.sharing_principle].append(item)
    standalone_by_principle: dict[str, list[_PlannedMethod]] = {p: [] for p in present_principles}
    for item in standalone:
        standalone_by_principle[item.sharing_principle].append(item)

    counters: dict[str, int] = {principle: 1 for principle in present_principles}
    second_step_groups: dict[str, dict[tuple[str, str | None], list[_PlannedMethod]]] = {}
    for principle in present_principles:
        first_items = first_by_principle[principle]
        standalone_items = standalone_by_principle[principle]
        first_direct = [item for item in first_items if item.subprinciple is None]
        first_by_sub: dict[str, list[_PlannedMethod]] = {}
        for item in first_items:
            if item.subprinciple is not None:
                first_by_sub.setdefault(item.subprinciple, []).append(item)
        first_child_count = len(first_direct) + len(first_by_sub)
        standalone_count = len(standalone_items)
        use_bucket_nodes = is_l2_fu and first_child_count > 0 and standalone_count > 0
        total_child_count = first_child_count + standalone_count
        first_parent = principle
        standalone_parent = principle
        if use_bucket_nodes:
            first_total = 0.5
            standalone_total = 0.5
            first_parent = _node_id_with_counter(
                label="m_s",
                counter_key="m_s",
                counters=counters,
            )
            standalone_parent = _node_id_with_counter(
                label="o_s",
                counter_key="o_s",
                counters=counters,
            )
            add(
                parent_id=principle,
                node_id=first_parent,
                label="m_s",
                node_type="principle",
                edge_weight=first_total,
                level=2,
            )
            add(
                parent_id=principle,
                node_id=standalone_parent,
                label="o_s",
                node_type="principle",
                edge_weight=standalone_total,
                level=2,
            )
            first_weight = 1.0 / first_child_count
            standalone_weight = 1.0 / standalone_count
        else:
            first_weight = 1.0 / total_child_count if first_child_count else 0.0
            standalone_weight = 1.0 / total_child_count if standalone_count else 0.0

        _add_first_stage_nodes(
            add=add,
            parent_id=first_parent,
            first_direct=first_direct,
            first_by_sub=first_by_sub,
            first_weight=first_weight,
            counters=counters,
            second_by_l1=second_by_l1,
            second_step_groups=second_step_groups,
        )
        for item in standalone_items:
            add(
                parent_id=standalone_parent,
                node_id=_node_id_with_counter(
                    label=item.enacting_metric,
                    counter_key=item.enacting_metric,
                    counters=counters,
                ),
                label=item.enacting_metric,
                node_type="metric",
                edge_weight=standalone_weight,
                level=4,
                candidate_label=item.candidate_label,
            )

    for first_metric_id, grouped in second_step_groups.items():
        if grouped:
            group_weight = 1.0 / len(grouped)
            group_node_ids: dict[tuple[str, str | None], str] = {}
            for sharing_principle, subprinciple in grouped:
                group_id = _node_id_with_counter(
                    label=sharing_principle,
                    counter_key=sharing_principle,
                    counters=counters,
                )
                add(
                    parent_id=first_metric_id,
                    node_id=group_id,
                    label=sharing_principle,
                    node_type="principle",
                    edge_weight=group_weight,
                    level=3,
                )
                group_node_ids[(sharing_principle, subprinciple)] = group_id
            for key, second_items in grouped.items():
                metric_weight = 1.0 / len(second_items)
                for second in second_items:
                    add(
                        parent_id=group_node_ids[key],
                        node_id=_node_id_with_counter(
                            label=second.enacting_metric,
                            counter_key=second.enacting_metric,
                            counters=counters,
                        ),
                        label=second.enacting_metric,
                        node_type="metric",
                        edge_weight=metric_weight,
                        level=4,
                        candidate_label=second.candidate_label,
                    )
    return rows


def _add_first_stage_nodes(
    *,
    add,
    parent_id: str,
    first_direct: list[_PlannedMethod],
    first_by_sub: dict[str, list[_PlannedMethod]],
    first_weight: float,
    counters: dict[str, int],
    second_by_l1: dict[str, list[_PlannedMethod]],
    second_step_groups: dict[str, dict[tuple[str, str | None], list[_PlannedMethod]]],
) -> None:
    for item in first_direct:
        metric_id = _node_id_with_counter(
            label=item.enacting_metric,
            counter_key=item.enacting_metric,
            counters=counters,
        )
        add(
            parent_id=parent_id,
            node_id=metric_id,
            label=item.enacting_metric,
            node_type="metric",
            edge_weight=first_weight,
            level=2,
            candidate_label=item.candidate_label,
        )
        second_step_groups[metric_id] = _group_second_steps(second_by_l1.get(item.method_name, []))
    for subprinciple, sub_items in first_by_sub.items():
        sub_id = _node_id_with_counter(
            label=subprinciple,
            counter_key=subprinciple,
            counters=counters,
        )
        add(
            parent_id=parent_id,
            node_id=sub_id,
            label=subprinciple,
            node_type="principle",
            edge_weight=first_weight,
            level=2,
        )
        sub_weight = 1.0 / len(sub_items)
        for item in sub_items:
            metric_id = _node_id_with_counter(
                label=item.enacting_metric,
                counter_key=item.enacting_metric,
                counters=counters,
            )
            add(
                parent_id=sub_id,
                node_id=metric_id,
                label=item.enacting_metric,
                node_type="metric",
                edge_weight=sub_weight,
                level=2,
                candidate_label=item.candidate_label,
            )
            second_step_groups[metric_id] = _group_second_steps(
                second_by_l1.get(item.method_name, [])
            )


def _first_stage_methods(
    *,
    candidates: tuple[InterMethodCandidate, ...],
) -> list[_PlannedMethod]:
    methods: list[_PlannedMethod] = []
    for candidate in candidates:
        if candidate.level == "level_1":
            methods.append(
                _describe_method(
                    method_name=str(candidate.l1_method),
                    candidate_label=candidate.candidate_label,
                )
            )
        elif candidate.l1_method is not None:
            methods.append(_describe_method(method_name=candidate.l1_method, candidate_label=None))
    return _unique_methods(methods)


def _standalone_methods(
    *,
    candidates: tuple[InterMethodCandidate, ...],
) -> list[_PlannedMethod]:
    return [
        _describe_method(
            method_name=str(candidate.l2_method),
            candidate_label=candidate.candidate_label,
        )
        for candidate in candidates
        if candidate.level == "level_2" and candidate.l1_method is None
    ]


def _second_step_methods_by_l1(
    *,
    candidates: tuple[InterMethodCandidate, ...],
) -> dict[str, list[_PlannedMethod]]:
    out: dict[str, list[_PlannedMethod]] = {}
    for candidate in candidates:
        if candidate.level == "level_2" and candidate.l1_method is not None:
            out.setdefault(candidate.l1_method, []).append(
                _describe_method(
                    method_name=str(candidate.l2_method),
                    candidate_label=candidate.candidate_label,
                )
            )
    return out


def _group_second_steps(
    second_steps: list[_PlannedMethod],
) -> dict[tuple[str, str | None], list[_PlannedMethod]]:
    grouped: dict[tuple[str, str | None], list[_PlannedMethod]] = {}
    for second in second_steps:
        key = (second.sharing_principle, second.subprinciple)
        grouped.setdefault(key, []).append(second)
    return grouped


def _describe_method(*, method_name: str, candidate_label: str | None) -> _PlannedMethod:
    sharing_principle, subprinciple, enacting_metric = parse_raw_asocc_method_label(method_name)
    return _PlannedMethod(
        method_name=str(method_name),
        sharing_principle=sharing_principle,
        subprinciple=subprinciple,
        enacting_metric=enacting_metric,
        candidate_label=candidate_label,
    )


def _unique_methods(methods: list[_PlannedMethod]) -> list[_PlannedMethod]:
    by_name: dict[str, _PlannedMethod] = {}
    for method in methods:
        if method.method_name not in by_name:
            by_name[method.method_name] = method
    return list(by_name.values())


def _sanitize_node_id(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace(",", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace("-", "_")
    )


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _node_id_with_counter(label: str, counter_key: str, counters: dict[str, int]) -> str:
    base = _sanitize_node_id(label)
    counters[counter_key] = counters.get(counter_key, 0) + 1
    index = counters[counter_key]
    if index == 1:
        return base
    return f"{base}_{index}"


def _text_or_none(value: object) -> str | None:
    if is_display_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _edge_weights_from_leaf_probabilities(
    *,
    frame: pd.DataFrame,
    leaf_probabilities: dict[str, float],
) -> list[float]:
    children = _children_by_parent(frame=frame)
    node_probability = _subtree_probabilities(frame=frame, leaf_probabilities=leaf_probabilities)
    weights: list[float] = []
    for parent_id, node_id in frame.loc[:, ["parent_id", "node_id"]].itertuples(
        index=False,
        name=None,
    ):
        parent_total = node_probability[str(parent_id)]
        if parent_total == 0.0:
            weights.append(1.0 / len(children[str(parent_id)]))
            continue
        weights.append(node_probability[str(node_id)] / parent_total)
    return weights


def _subtree_probabilities(
    *,
    frame: pd.DataFrame,
    leaf_probabilities: dict[str, float],
) -> dict[str, float]:
    children = _children_by_parent(frame=frame)
    leaf_by_node = {
        str(node_id): str(label)
        for node_id, label in frame.loc[:, ["node_id", "candidate_label"]].itertuples(
            index=False,
            name=None,
        )
        if _text_or_none(label) is not None
    }
    totals: dict[str, float] = {}

    def total(node_id: str) -> float:
        if node_id in leaf_by_node:
            totals[node_id] = float(leaf_probabilities[leaf_by_node[node_id]])
            return totals[node_id]
        totals[node_id] = sum(total(child_id) for child_id in children[node_id])
        return totals[node_id]

    total("root")
    return totals


def _children_by_parent(*, frame: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for parent_id, node_id in frame.loc[:, ["parent_id", "node_id"]].itertuples(
        index=False,
        name=None,
    ):
        children[str(parent_id)].append(str(node_id))
    return children


def _leaf_probabilities_from_edges(*, frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.copy()
    frame["edge_weight"] = pd.to_numeric(frame["edge_weight"], errors="raise")
    _validate_parent_edge_sums(frame=frame)
    children = _children_by_parent(frame=frame)
    row_by_node = {
        str(node_id): (float(edge_weight), candidate_label)
        for _parent_id, node_id, _label, _node_type, edge_weight, _level, candidate_label in (
            frame.loc[:, list(INTER_METHOD_TREE_COLUMNS)].itertuples(index=False, name=None)
        )
    }
    probabilities: dict[str, float] = {}

    def walk(node_id: str, probability: float) -> None:
        for child_id in children.get(node_id, []):
            edge_weight, raw_candidate_label = row_by_node[child_id]
            next_probability = probability * edge_weight
            candidate_label = _text_or_none(raw_candidate_label)
            if candidate_label is None:
                walk(child_id, next_probability)
                continue
            probabilities[candidate_label] = next_probability

    walk("root", 1.0)
    return probabilities


def _validate_tree_topology(*, expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    compare_columns = [column for column in INTER_METHOD_TREE_COLUMNS if column != "edge_weight"]
    expected_compare = expected.loc[:, compare_columns].fillna("").reset_index(drop=True)
    actual_compare = actual.loc[:, compare_columns].fillna("").reset_index(drop=True)
    if not expected_compare.equals(actual_compare):
        raise ValueError(
            "Inter-method tree CSV topology does not match the current method scope. "
            "Regenerate the weight CSV for the requested base_asocc_args and external "
            "method scope before editing edge weights."
        )
    actual["edge_weight"] = pd.to_numeric(actual["edge_weight"], errors="raise")
    if bool(((actual["edge_weight"] < 0.0) | (actual["edge_weight"] > 1.0)).any()):
        raise ValueError("Inter-method tree edge weights must be between 0 and 1.")
    _validate_parent_edge_sums(frame=actual)


def _validate_parent_edge_sums(*, frame: pd.DataFrame) -> None:
    sums = frame.groupby("parent_id", dropna=False)["edge_weight"].sum()
    values = np.asarray(sums, dtype=np.float64)
    if bool((np.abs(values - 1.0) > EDGE_WEIGHT_TOLERANCE).any()):
        raise ValueError("Inter-method sibling edge weights must sum to 1 for each parent.")
