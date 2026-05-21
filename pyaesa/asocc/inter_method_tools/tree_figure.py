"""Figure rendering for aSoCC inter-method probability trees."""

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from pyaesa.shared.figures.save import create_figure, save_figure

L1_COLOR = "#E8E5F2"
L2_COLOR = "#DCECD6"
EDGE_COLOR = "#7A8088"
TEXT_COLOR = "#4B5563"
LEGEND_FONT_SIZE = 10
LEGEND_ICON_LABEL_GAP = 1
LEGEND_ITEM_GAP = 7

PRINCIPLE_RADIUS = 0.28
L1_METRIC_HALF_WIDTH = 1.35
L2_METRIC_HALF_WIDTH = 0.9
METRIC_HEIGHT = 0.52

X_START = 0.8
X_L1_PRINCIPLE = 2.4
X_L1_STEP_BUCKET = 3.6
X_L1_SUBPRINCIPLE = 4.7
X_L1_METRIC = 7.0
X_L2_PRINCIPLE = 10.2
X_L2_METRIC = 12.8
Y_LEAF_GAP = 0.8


def render_inter_method_tree(
    *,
    frame: pd.DataFrame,
    figure_base_path: Path,
    output_format: str,
    dpi: int,
) -> list[Path]:
    """Render one inter-method probability tree."""
    plot_frame = frame.copy()
    plot_frame["edge_weight"] = pd.Series(
        pd.to_numeric(pd.Series(plot_frame.loc[:, "edge_weight"], copy=False), errors="raise"),
        index=plot_frame.index,
    )
    plot_frame["level"] = pd.Series(
        pd.to_numeric(pd.Series(plot_frame.loc[:, "level"], copy=False), errors="raise"),
        index=plot_frame.index,
    ).astype("int64")
    children, row_by_node = _build_maps(frame=plot_frame)
    terminals = _terminal_nodes(children=children)
    y_positions = {
        terminal: float(len(terminals) - index - 1) * Y_LEAF_GAP
        for index, terminal in enumerate(terminals)
    }
    _assign_internal_y(node_id="root", children=children, y_positions=y_positions)
    coords, kinds, extents = _node_layout(row_by_node=row_by_node, y_positions=y_positions)
    cumulative = _cumulative_probabilities(children=children, row_by_node=row_by_node)
    final_x = (
        X_L2_METRIC + L2_METRIC_HALF_WIDTH + 0.75
        if any(kind in {"l2_principle", "l2_metric"} for kind in kinds.values())
        else X_L1_METRIC + L1_METRIC_HALF_WIDTH + 0.75
    )

    fig = create_figure(figsize=(16.0, max(4.0, len(terminals) * 0.88 + 1.6)))
    ax = fig.subplots()
    ax.set_facecolor("white")

    start_y = y_positions["root"]
    start_patch = _render_start(ax=ax, x=X_START, y=start_y)
    patches: dict[str, object] = {
        node_id: _render_node(
            ax=ax,
            x=coords[node_id][0],
            y=coords[node_id][1],
            row=row,
            kind=kinds[node_id],
        )
        for node_id, row in row_by_node.items()
    }
    _render_edges(
        ax=ax,
        parent_id="root",
        children=children,
        row_by_node=row_by_node,
        coords=coords,
        kinds=kinds,
        extents=extents,
        patches=patches,
        start_patch=start_patch,
        start_y=start_y,
    )
    for terminal in terminals:
        x, y = coords[terminal]
        _render_horizontal_arrow(
            ax=ax,
            x1=x + extents[terminal],
            y=y,
            x2=final_x,
        )
        ax.text(
            final_x + 0.05,
            y,
            _percent(cumulative[terminal]),
            ha="left",
            va="center",
            fontsize=12,
            zorder=4,
        )

    tree_bottom = _tree_bottom(coords=coords, row_by_node=row_by_node, start_y=start_y)
    tree_top = _tree_top(coords=coords, row_by_node=row_by_node, start_y=start_y)
    legend_bounds = _render_legend(
        ax=ax,
        y_top=tree_bottom - 0.12,
        x_center=(X_START + final_x) / 2,
        include_step_buckets=any(
            str(row["label"]) in {"m_s", "o_s"} for row in row_by_node.values()
        ),
    )
    min_x = min(
        [
            X_START - 0.4,
            legend_bounds[0],
            *(x - extents[n] for n, (x, _y) in coords.items()),
        ]
    )
    max_x = max(
        [
            final_x + 1.05,
            legend_bounds[2],
            *(x + extents[n] for n, (x, _y) in coords.items()),
        ]
    )
    ax.set_xlim(min_x - 0.3, max_x + 0.3)
    ax.set_ylim(legend_bounds[1] - 0.18, tree_top + 0.22)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.02)
    return save_figure(
        fig,
        Path(figure_base_path),
        dpi=dpi,
        output_format=output_format,
    )


def _build_maps(
    *,
    frame: pd.DataFrame,
) -> tuple[dict[str, list[str]], dict[str, dict[str, object]]]:
    children: dict[str, list[str]] = defaultdict(list)
    row_by_node: dict[str, dict[str, object]] = {}
    for row in frame.to_dict(orient="records"):
        parent_id = str(row["parent_id"])
        node_id = str(row["node_id"])
        children[parent_id].append(node_id)
        row_by_node[node_id] = row
    return dict(children), row_by_node


def _terminal_nodes(*, children: dict[str, list[str]]) -> list[str]:
    terminals: list[str] = []

    def walk(node_id: str) -> None:
        child_ids = children.get(node_id, [])
        if child_ids:
            for child_id in child_ids:
                walk(child_id)
            return
        terminals.append(node_id)

    for root_child_id in children["root"]:
        walk(root_child_id)
    return terminals


def _assign_internal_y(
    *,
    node_id: str,
    children: dict[str, list[str]],
    y_positions: dict[str, float],
) -> float:
    if node_id in y_positions:
        return y_positions[node_id]
    child_y = [
        _assign_internal_y(node_id=child_id, children=children, y_positions=y_positions)
        for child_id in children[node_id]
    ]
    y_positions[node_id] = sum(child_y) / len(child_y)
    return y_positions[node_id]


def _node_layout(
    *,
    row_by_node: dict[str, dict[str, object]],
    y_positions: dict[str, float],
) -> tuple[dict[str, tuple[float, float]], dict[str, str], dict[str, float]]:
    coords = {"root": (X_START, y_positions["root"])}
    kinds: dict[str, str] = {}
    extents: dict[str, float] = {"root": 0.4}
    for node_id, row in row_by_node.items():
        kind = _node_kind(row=row)
        kinds[node_id] = kind
        extents[node_id] = _node_extent(row=row, kind=kind)
        coords[node_id] = (_x_for_kind(kind), y_positions[node_id])
    return coords, kinds, extents


def _node_kind(*, row: dict[str, object]) -> str:
    level = int(cast(int, row["level"]))
    node_type = str(row["node_type"]).lower()
    if node_type == "principle":
        if str(row["label"]) in {"m_s", "o_s"}:
            return "step_bucket"
        if level == 1:
            return "l1_principle"
        if level == 2:
            return "l1_subprinciple"
        return "l2_principle"
    if level == 4:
        return "l2_metric"
    return "l1_metric"


def _x_for_kind(kind: str) -> float:
    return {
        "l1_principle": X_L1_PRINCIPLE,
        "step_bucket": X_L1_STEP_BUCKET,
        "l1_subprinciple": X_L1_SUBPRINCIPLE,
        "l1_metric": X_L1_METRIC,
        "l2_principle": X_L2_PRINCIPLE,
        "l2_metric": X_L2_METRIC,
    }[kind]


def _node_extent(*, row: dict[str, object], kind: str) -> float:
    if str(row["node_type"]).lower() == "principle":
        return PRINCIPLE_RADIUS
    if kind == "l2_metric":
        return L2_METRIC_HALF_WIDTH
    return L1_METRIC_HALF_WIDTH


def _cumulative_probabilities(
    *,
    children: dict[str, list[str]],
    row_by_node: dict[str, dict[str, object]],
) -> dict[str, float]:
    cumulative = {"root": 1.0}

    def walk(node_id: str) -> None:
        for child_id in children.get(node_id, []):
            cumulative[child_id] = cumulative[node_id] * float(
                cast(float, row_by_node[child_id]["edge_weight"])
            )
            walk(child_id)

    walk("root")
    return cumulative


def _render_start(*, ax, x: float, y: float):
    width = 0.8
    height = 0.5
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="square,pad=0.02",
        facecolor="white",
        edgecolor="black",
        lw=1.2,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, "Start", ha="center", va="center", fontsize=12, zorder=4)
    return patch


def _render_node(*, ax, x: float, y: float, row: dict[str, object], kind: str):
    label = str(row["label"])
    if str(row["node_type"]).lower() == "principle":
        patch = Circle(
            (x, y),
            PRINCIPLE_RADIUS,
            facecolor=_node_fill_color(kind=kind),
            edgecolor="black",
            lw=1.2,
            linestyle=(0, (2, 4)) if kind in {"l1_subprinciple", "step_bucket"} else "solid",
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            style="italic" if kind in {"l1_subprinciple", "step_bucket"} else "normal",
            zorder=4,
        )
        return patch
    width = 2 * (L2_METRIC_HALF_WIDTH if kind == "l2_metric" else L1_METRIC_HALF_WIDTH)
    patch = FancyBboxPatch(
        (x - width / 2, y - METRIC_HEIGHT / 2),
        width,
        METRIC_HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        facecolor=_node_fill_color(kind=kind),
        edgecolor="black",
        lw=1.2,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=12, zorder=4)
    return patch


def _node_fill_color(*, kind: str) -> str:
    return L2_COLOR if kind in {"l2_principle", "l2_metric"} else L1_COLOR


def _render_edges(
    *,
    ax,
    parent_id: str,
    children: dict[str, list[str]],
    row_by_node: dict[str, dict[str, object]],
    coords: dict[str, tuple[float, float]],
    kinds: dict[str, str],
    extents: dict[str, float],
    patches: dict[str, object],
    start_patch,
    start_y: float,
) -> None:
    child_ids = children.get(parent_id, [])
    if not child_ids:
        return
    if parent_id == "root":
        parent_x = X_START + 0.4
        parent_y = start_y
        parent_kind = "root"
    else:
        parent_x = coords[parent_id][0] + extents[parent_id]
        parent_y = coords[parent_id][1]
        parent_kind = kinds[parent_id]
    split_x = parent_x + _split_offset(kind=parent_kind, child_count=len(child_ids))
    if len(child_ids) > 1:
        child_y_values = [coords[child_id][1] for child_id in child_ids]
        ax.plot([parent_x, split_x], [parent_y, parent_y], color=EDGE_COLOR, lw=1.2, zorder=1)
        ax.plot(
            [split_x, split_x],
            [min(child_y_values), max(child_y_values)],
            color=EDGE_COLOR,
            lw=1.2,
            zorder=1,
        )
    for child_id in child_ids:
        child_x = coords[child_id][0] - extents[child_id]
        child_y = coords[child_id][1]
        edge_start_x = split_x if len(child_ids) > 1 else parent_x
        edge_start_y = child_y if len(child_ids) > 1 else parent_y
        _render_horizontal_arrow(
            ax=ax,
            x1=edge_start_x,
            y=edge_start_y,
            x2=child_x,
            target_patch=patches[child_id],
            source_patch=(
                (start_patch if parent_id == "root" else patches[parent_id])
                if len(child_ids) == 1
                else None
            ),
        )
        _render_edge_label(
            ax=ax,
            x=edge_start_x,
            y=child_y if len(child_ids) > 1 else parent_y,
            x2=child_x,
            label=_percent(float(cast(float, row_by_node[child_id]["edge_weight"]))),
        )
        _render_edges(
            ax=ax,
            parent_id=child_id,
            children=children,
            row_by_node=row_by_node,
            coords=coords,
            kinds=kinds,
            extents=extents,
            patches=patches,
            start_patch=start_patch,
            start_y=start_y,
        )


def _split_offset(*, kind: str, child_count: int) -> float:
    if child_count == 1:
        return 0.0
    return {
        "root": 0.2,
        "l1_principle": 0.1,
        "l1_subprinciple": 0.1,
    }.get(kind, 0.45)


def _render_horizontal_arrow(
    *,
    ax,
    x1: float,
    y: float,
    x2: float,
    target_patch=None,
    source_patch=None,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=EDGE_COLOR,
            shrinkA=0,
            shrinkB=0,
            patchA=source_patch,
            patchB=target_patch,
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
    )


def _render_edge_label(*, ax, x: float, y: float, x2: float, label: str) -> None:
    label_x = (x + x2) / 2 if label == "100%" else x + 0.04
    ha = "center" if label == "100%" else "left"
    ax.text(
        label_x,
        y + 0.08,
        label,
        fontsize=10,
        color=TEXT_COLOR,
        ha=ha,
        va="center",
        style="italic",
        zorder=4,
    )


def _render_legend(
    *,
    ax,
    y_top: float,
    x_center: float,
    include_step_buckets: bool,
) -> tuple[float, float, float, float]:
    items: list[tuple[str, str, str]] = [
        ("circle", "", "Sharing principle"),
        ("metric", "", "Enacting metric"),
        ("square", L1_COLOR, "Level 1: Country"),
        ("square", L2_COLOR, "Level 2: Sector"),
    ]
    if include_step_buckets:
        items.append(("circle_dotted", "", "o_s/m_s: one and multi step allocation paths"))
    x = x_center
    y = y_top - 0.32
    frame_height = 0.44
    item_boxes = [
        _legend_item_box(item_type=item_type, color=color, label=label)
        for item_type, color, label in items
    ]
    packed = HPacker(
        children=cast(Any, item_boxes),
        align="center",
        pad=0,
        sep=LEGEND_ITEM_GAP,
    )
    legend = AnnotationBbox(
        packed,
        (x, y),
        xycoords="data",
        box_alignment=(0.5, 0.5),
        frameon=True,
        pad=0,
        bboxprops={
            "boxstyle": "round,pad=0.5,rounding_size=0.08",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.9,
        },
        annotation_clip=False,
        zorder=2,
    )
    legend.set_clip_on(False)
    ax.add_artist(legend)
    return x - 0.12, y - frame_height / 2, x + 0.12, y_top


def _legend_item_box(*, item_type: str, color: str, label: str) -> HPacker:
    icon = _legend_icon_box(item_type=item_type, color=color)
    text = TextArea(
        label,
        textprops={
            "fontsize": LEGEND_FONT_SIZE,
            "color": TEXT_COLOR,
            "style": "italic",
        },
    )
    return HPacker(
        children=cast(Any, [icon, text]),
        align="center",
        pad=0,
        sep=LEGEND_ICON_LABEL_GAP,
    )


def _legend_icon_box(*, item_type: str, color: str) -> DrawingArea:
    if item_type in {"circle", "circle_dotted"}:
        area = DrawingArea(17, 17, 0, 0)
        area.add_artist(
            Circle(
                (8.5, 8.5),
                6.8,
                facecolor="white",
                edgecolor="black",
                lw=0.8,
                linestyle=(0, (2, 4)) if item_type == "circle_dotted" else "solid",
            )
        )
        return area
    if item_type == "metric":
        area = DrawingArea(44, 17, 0, 0)
        area.add_artist(
            FancyBboxPatch(
                (1.0, 3.5),
                42.0,
                10.0,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                facecolor="white",
                edgecolor="black",
                lw=0.8,
            )
        )
        return area
    area = DrawingArea(17, 17, 0, 0)
    area.add_artist(
        Rectangle(
            (4.0, 4.0),
            9.0,
            9.0,
            facecolor=color,
            edgecolor="black",
            lw=0.8,
        )
    )
    return area


def _tree_bottom(
    *,
    coords: dict[str, tuple[float, float]],
    row_by_node: dict[str, dict[str, object]],
    start_y: float,
) -> float:
    bottoms = [start_y - 0.25]
    for node_id, row in row_by_node.items():
        bottoms.append(coords[node_id][1] - _vertical_extent(row=row))
    return min(bottoms)


def _tree_top(
    *,
    coords: dict[str, tuple[float, float]],
    row_by_node: dict[str, dict[str, object]],
    start_y: float,
) -> float:
    tops = [start_y + 0.25]
    for node_id, row in row_by_node.items():
        tops.append(coords[node_id][1] + _vertical_extent(row=row))
    return max(tops)


def _vertical_extent(*, row: dict[str, object]) -> float:
    if str(row["node_type"]).lower() == "principle":
        return PRINCIPLE_RADIUS
    return METRIC_HEIGHT / 2


def _percent(value: float) -> str:
    pct = round(float(value) * 100.0, 2)
    text = f"{int(pct)}" if pct.is_integer() else f"{pct:.2f}".rstrip("0").rstrip(".")
    return f"{text}%"
