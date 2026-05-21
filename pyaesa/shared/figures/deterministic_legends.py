"""Shared grouped legend helpers for deterministic comparison figures."""

from collections import OrderedDict
from collections.abc import Mapping

from pyaesa.shared.figures.figure_footer import (
    _LEGEND_ROW_HEIGHT_IN,
    _LEGEND_TITLED_OVERHEAD_IN,
    align_lower_legend_stack_top_to_layout,
    align_lower_legend_top_to_layout,
    center_legend_text,
    _legend_columns,
    grouped_entries_from_figure,
    legend_note_entries,
    legend_display_rows,
    reserve_footer_space,
)

_ORDER = [
    "Consumption-anchored two steps",
    "Production-anchored two steps",
    "Two-step",
    "One-step",
]


def render_grouped_deterministic_legend_below(
    axis,
    *,
    legend_note: str | None = None,
    handler_map: Mapping[type, object] | None = None,
    legend_kwargs: Mapping[str, object] | None = None,
    legend_kwargs_group_title: str | None = None,
    legend_extra_height_in: float = 0.0,
    extra_height_in: float = 0.0,
    hidden_group_titles: set[str] | None = None,
    excluded_group_titles: set[str] | None = None,
    anchor_x: float | None = None,
) -> None:
    """Render grouped deterministic legends below the full figure."""
    fig = axis.figure
    grouped = _ordered_groups(grouped_entries_from_figure(fig))
    note_entries = legend_note_entries(legend_note)
    if excluded_group_titles:
        grouped = OrderedDict(
            (title, entries)
            for title, entries in grouped.items()
            if title not in excluded_group_titles
        )
    if not grouped and not note_entries:
        return
    legend_x = 0.5 if anchor_x is None else float(anchor_x)
    extra_legend_kwargs = dict(legend_kwargs or {})
    deduped: OrderedDict[str, list[tuple]] = OrderedDict()
    seen_labels: set[str] = set()
    for title, entries in grouped.items():
        unique_entries: list[tuple] = []
        for handle, label in entries:
            if label in seen_labels:
                continue
            unique_entries.append((handle, label))
            seen_labels.add(label)
        if unique_entries:
            deduped[title] = unique_entries
    grouped = deduped
    if len(grouped) <= 1 and not note_entries:
        title, entries = next(iter(grouped.items()))
        visible_title = _visible_title(title, hidden_group_titles=hidden_group_titles)
        ncol = _legend_columns(entry_count=len(entries), max_columns=5)
        rows = legend_display_rows([label for _handle, label in entries], ncol=ncol)
        layout = reserve_footer_space(
            fig,
            rows=rows,
            note_lines=0,
            title_rows=1 if visible_title else 0,
            extra_height_in=(
                _legend_extra_height(
                    title,
                    group_title=legend_kwargs_group_title,
                    extra_height_in=legend_extra_height_in,
                )
                + float(extra_height_in)
            ),
        )
        legend = fig.legend(
            handles=[handle for handle, _label in entries],
            labels=[label for _handle, label in entries],
            loc="lower center",
            bbox_to_anchor=(legend_x, layout.anchor_y),
            ncol=ncol,
            frameon=False,
            fontsize="small",
            title=visible_title,
            title_fontsize="small",
            handler_map=handler_map,
            **_legend_kwargs(
                title,
                group_title=legend_kwargs_group_title,
                extra_legend_kwargs=extra_legend_kwargs,
            ),
        )
        if visible_title:
            legend.get_title().set_fontweight("bold")
        align_lower_legend_top_to_layout(fig, legend, layout=layout, anchor_x=legend_x)
        return
    group_specs = []
    total_rows = 0
    title_rows = 0
    total_extra_height_in = 0.0
    if note_entries:
        rows = legend_display_rows([label for _handle, label in note_entries], ncol=1)
        total_rows += rows
        group_specs.append(("", "", note_entries, 1, rows, 0.0))
    for title, entries in grouped.items():
        visible_title = _visible_title(title, hidden_group_titles=hidden_group_titles)
        if visible_title:
            title_rows += 1
        ncol = _legend_columns(entry_count=len(entries), max_columns=5)
        rows = legend_display_rows([label for _handle, label in entries], ncol=ncol)
        total_rows += rows
        group_extra_height_in = _legend_extra_height(
            title,
            group_title=legend_kwargs_group_title,
            extra_height_in=legend_extra_height_in,
        )
        total_extra_height_in += group_extra_height_in
        group_specs.append((title, visible_title, entries, ncol, rows, group_extra_height_in))
    layout = reserve_footer_space(
        fig,
        rows=total_rows,
        note_lines=0,
        title_rows=title_rows,
        extra_height_in=(total_extra_height_in + float(extra_height_in)),
    )
    fig_height = fig.get_size_inches()[1]
    anchor_y = layout.anchor_y
    legends = []
    for title, visible_title, entries, ncol, rows, group_extra_height_in in group_specs:
        legend_options = _legend_kwargs(
            title,
            group_title=legend_kwargs_group_title,
            extra_legend_kwargs=extra_legend_kwargs,
        )
        if entries == note_entries:
            legend_options = {**legend_options, "handlelength": 0, "handletextpad": 0}
        legend = fig.legend(
            handles=[handle for handle, _label in entries],
            labels=[label for _handle, label in entries],
            title=visible_title,
            loc="lower center",
            bbox_to_anchor=(legend_x, anchor_y),
            ncol=ncol,
            fontsize="small",
            title_fontsize="small",
            frameon=False,
            handler_map=handler_map,
            **legend_options,
        )
        if visible_title:
            legend.get_title().set_fontweight("bold")
        if entries == note_entries:
            center_legend_text(legend)
        fig.add_artist(legend)
        legends.append(legend)
        overhead_in = _LEGEND_TITLED_OVERHEAD_IN if visible_title else 0.0
        anchor_y += (
            _LEGEND_ROW_HEIGHT_IN * rows + overhead_in + group_extra_height_in
        ) / fig_height
    align_lower_legend_stack_top_to_layout(fig, legends, layout=layout, anchor_x=legend_x)


def bind_deterministic_legend_group(handle, group_title: str) -> None:
    """Attach one deterministic legend group title to a rendered artist."""
    setattr(handle, "_pyaesa_group_title", str(group_title).strip())


def _ordered_groups(
    grouped: OrderedDict[str, list[tuple]],
) -> OrderedDict[str, list[tuple]]:
    ordered: OrderedDict[str, list[tuple]] = OrderedDict()
    for title in _ORDER:
        if title in grouped and grouped[title]:
            ordered[title] = grouped[title]
    for title, items in grouped.items():
        if title not in ordered and items:
            ordered[title] = items
    return ordered


def _legend_kwargs(
    title: str,
    *,
    group_title: str | None,
    extra_legend_kwargs: dict[str, object],
) -> dict[str, object]:
    if not extra_legend_kwargs:
        return {}
    if group_title is None or title == group_title:
        return extra_legend_kwargs
    return {}


def _legend_extra_height(
    title: str,
    *,
    group_title: str | None,
    extra_height_in: float,
) -> float:
    if extra_height_in <= 0.0:
        return 0.0
    if group_title is None or title == group_title:
        return float(extra_height_in)
    return 0.0


def _visible_title(title: str, *, hidden_group_titles: set[str] | None) -> str:
    if hidden_group_titles is not None and title in hidden_group_titles:
        return ""
    return title
