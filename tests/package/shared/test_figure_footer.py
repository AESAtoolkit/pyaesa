from collections import OrderedDict

import matplotlib.pyplot as plt

from pyaesa.shared.figures import figure_footer as footer_mod


def test_legend_note_and_label_row_contracts() -> None:
    fig, _ax = plt.subplots()

    assert footer_mod.legend_note_lines(fig, None) == []
    assert footer_mod.legend_note_lines(fig, " \n  ") == []
    assert footer_mod.legend_note_lines(fig, "line one\n\nline two") == [
        "line one",
        "line two",
    ]
    wrapped = footer_mod.legend_note_lines(
        fig,
        "A very long deterministic variant compression note that must wrap within "
        "the plotting area width.",
    )
    assert len(wrapped) > 1
    assert " ".join(wrapped) == (
        "A very long deterministic variant compression note that must wrap within "
        "the plotting area width."
    )
    assert footer_mod.legend_label_line_count(" \n  ") == 0
    assert footer_mod.legend_label_line_count("method\nplain min: 2020\n\ndotted max: 2030") == 3
    assert (
        footer_mod.legend_display_rows(
            ["method\nplain min: 2020\ndotted max: 2030", "plain", "second\nline"],
            ncol=2,
        )
        == 5
    )
    assert footer_mod.legend_display_rows([], ncol=3) == 0

    plt.close(fig)


def test_render_below_figure_legend_handles_empty_and_duplicate_labels() -> None:
    empty_fig, _empty_ax = plt.subplots()
    assert footer_mod.render_below_figure_legend(empty_fig) is False
    plt.close(empty_fig)

    duplicate_fig, duplicate_ax = plt.subplots()
    duplicate_ax.plot([2030, 2031], [1.0, 2.0], label="same")
    duplicate_ax.plot([2030, 2031], [2.0, 3.0], label="same")

    assert footer_mod.render_below_figure_legend(duplicate_fig) is True
    assert [text.get_text() for text in duplicate_fig.legends[0].texts] == ["same"]

    plt.close(duplicate_fig)

    fig, ax = plt.subplots()
    ax.plot([2030, 2031], [1.0, 2.0], label="one")
    ax.plot([2030, 2031], [2.0, 3.0], label="two")

    assert footer_mod.render_below_figure_legend(fig, legend_note="note", max_columns=2) is True
    assert len(fig.legends) == 2
    assert [text.get_text() for text in fig.legends[0].texts] == ["note"]
    assert [text.get_text() for text in fig.legends[1].texts] == ["one", "two"]

    plt.close(fig)

    note_only_fig, _note_only_ax = plt.subplots()
    assert footer_mod.render_below_figure_legend(note_only_fig, legend_note="note") is True
    assert [text.get_text() for text in note_only_fig.legends[0].texts] == ["note"]
    plt.close(note_only_fig)

    multi_axis_fig, (left_ax, right_ax) = plt.subplots(1, 2)
    left_ax.plot([2030, 2031], [0.0, 0.0], label=" ")
    left_ax.plot([2030, 2031], [1.0, 2.0], label="shared")
    right_ax.plot([2030, 2031], [2.0, 3.0], label="shared")

    assert footer_mod.render_below_figure_legend(multi_axis_fig) is True
    assert [text.get_text() for text in multi_axis_fig.legends[0].texts] == ["shared"]

    plt.close(multi_axis_fig)

    extra_fig, extra_ax = plt.subplots()
    extra_handle = extra_ax.plot([2030, 2031], [1.0, 2.0], label="_nolegend_")[0]
    assert (
        footer_mod.render_below_figure_legend(
            extra_fig,
            extra_entries=[
                (extra_handle, "extra"),
                (extra_handle, " "),
                (extra_handle, "extra"),
            ],
        )
        is True
    )
    assert [text.get_text() for text in extra_fig.legends[0].texts] == ["extra"]
    plt.close(extra_fig)

    panel_fig, (left_axis, right_axis) = plt.subplots(1, 2)
    left_handle = left_axis.plot([2030, 2031], [1.0, 2.0], label="left")[0]
    footer_mod.render_two_panel_legends_below(
        panel_fig,
        left_axis=left_axis,
        right_axis=right_axis,
        left_handles=[left_handle],
        right_handles=[],
        left_ncol=1,
        right_ncol=1,
    )
    assert [text.get_text() for text in panel_fig.legends[0].texts] == ["left"]
    plt.close(panel_fig)

    right_only_fig, (left_axis, right_axis) = plt.subplots(1, 2)
    right_handle = right_axis.plot([2030, 2031], [1.0, 2.0], label="right")[0]
    footer_mod.render_two_panel_legends_below(
        right_only_fig,
        left_axis=left_axis,
        right_axis=right_axis,
        left_handles=[],
        right_handles=[right_handle],
        left_ncol=1,
        right_ncol=1,
    )
    assert [text.get_text() for text in right_only_fig.legends[0].texts] == ["right"]
    plt.close(right_only_fig)


def test_grouped_entries_and_footer_layout_cover_group_validation() -> None:
    fig, ax = plt.subplots()
    first_line = ax.plot([2030, 2031], [1.0, 2.0], label="A")[0]
    second_line = ax.plot([2030, 2031], [2.0, 3.0], label="B")[0]
    footer_mod.reserve_footer_space(fig, rows=1, note_lines=1, title_rows=1)
    first_line._pyaesa_group_title = "First"  # type: ignore[attr-defined]
    second_line._pyaesa_group_title = "Second"  # type: ignore[attr-defined]

    grouped = footer_mod.grouped_entries_from_figure(fig)

    assert grouped == OrderedDict(
        {
            "First": [(first_line, "A")],
            "Second": [(second_line, "B")],
        }
    )
    assert float(fig.subplotpars.bottom) > 0.0

    plt.close(fig)

    unclassified_fig, unclassified_ax = plt.subplots()
    unclassified_line = unclassified_ax.plot([2030, 2031], [1.0, 2.0], label="A")[0]
    unclassified_line._pyaesa_group_title = ""  # type: ignore[attr-defined]

    assert footer_mod.grouped_entries_from_figure(unclassified_fig) == OrderedDict(
        {"": [(unclassified_line, "A")]}
    )

    plt.close(unclassified_fig)

    missing_group_fig, missing_group_ax = plt.subplots()
    missing_group_line = missing_group_ax.plot([2030, 2031], [1.0, 2.0], label="A")[0]

    assert footer_mod.grouped_entries_from_figure(missing_group_fig) == OrderedDict(
        {"": [(missing_group_line, "A")]}
    )

    plt.close(missing_group_fig)


def test_footer_contracts_cover_blank_labels_and_layout_fallbacks() -> None:
    grouped_fig, grouped_ax = plt.subplots()
    grouped_ax.plot([2030, 2031], [1.0, 2.0], label=" ")
    grouped_entries = footer_mod.grouped_entries_from_figure(grouped_fig)
    assert grouped_entries == OrderedDict()
    plt.close(grouped_fig)

    collect_fig, collect_ax = plt.subplots()
    collect_ax.plot([2030, 2031], [1.0, 2.0], label=" ")
    collect_ax.plot([2030, 2031], [2.0, 3.0], label="kept")
    layout = footer_mod.reserve_footer_space(collect_fig, rows=0, note_lines=0)
    assert layout.bottom >= 0.0
    assert layout.anchor_y > 0.0
    assert float(collect_fig.get_size_inches()[1]) > 0.0
    plt.close(collect_fig)

    tall_fig, _tall_ax = plt.subplots(figsize=(4.0, 8.0))
    tall_layout = footer_mod.reserve_footer_space(tall_fig, rows=0, note_lines=0)
    assert float(tall_fig.get_size_inches()[1]) == 8.0
    assert tall_layout.bottom >= 0.0
    plt.close(tall_fig)

    min_height_fig, _min_height_ax = plt.subplots(figsize=(4.0, 8.0))
    footer_mod.set_footer_min_plot_height(min_height_fig, height_in=7.5)
    min_height_layout = footer_mod.reserve_footer_space(min_height_fig, rows=0, note_lines=0)
    assert float(min_height_fig.get_size_inches()[1]) >= 8.0
    assert min_height_layout.bottom >= 0.0
    plt.close(min_height_fig)

    resize_fig, _resize_ax = plt.subplots(figsize=(4.0, 2.0))
    resized_layout = footer_mod.reserve_footer_space(resize_fig, rows=3, note_lines=2, title_rows=2)
    assert float(resize_fig.get_size_inches()[1]) > 2.0
    assert resized_layout.bottom > 0.0
    plt.close(resize_fig)

    no_tick_fig, no_tick_ax = plt.subplots()
    no_tick_ax.set_xticks([])
    no_tick_ax.set_xlabel("")
    no_tick_layout = footer_mod.reserve_footer_space(no_tick_fig, rows=1, note_lines=0)
    assert no_tick_layout.bottom > 0.0
    plt.close(no_tick_fig)

    year_fig, year_ax = plt.subplots()
    year_ax.set_xlabel("Year")
    year_layout = footer_mod.reserve_footer_space(year_fig, rows=1, note_lines=0)
    assert year_layout.bottom > no_tick_layout.bottom
    plt.close(year_fig)

    rotated_fig, rotated_ax = plt.subplots()
    rotated_ax.set_xticks([0, 1])
    rotated_ax.set_xticklabels(["long rotated label", "second long rotated label"], rotation=60)
    rotated_layout = footer_mod.reserve_footer_space(rotated_fig, rows=1, note_lines=0)
    assert rotated_layout.bottom > no_tick_layout.bottom
    plt.close(rotated_fig)

    aligned_fig, aligned_ax = plt.subplots()
    aligned_line = aligned_ax.plot([2030, 2031], [1.0, 2.0], label="aligned")[0]
    aligned_legend = aligned_fig.legend(handles=[aligned_line], labels=["aligned"])
    aligned_layout = footer_mod.reserve_footer_space(aligned_fig, rows=1, note_lines=0)
    footer_mod.align_lower_legend_top_to_layout(aligned_fig, aligned_legend, layout=aligned_layout)
    assert aligned_fig.legends[0].get_visible()
    plt.close(aligned_fig)

    renderer_fig, renderer_ax = plt.subplots()
    renderer_line = renderer_ax.plot([2030, 2031], [1.0, 2.0], label="renderer")[0]
    renderer_legend = renderer_fig.legend(handles=[renderer_line], labels=["renderer"])
    renderer_layout = footer_mod.reserve_footer_space(renderer_fig, rows=0, note_lines=0)
    footer_mod.align_lower_legend_top_to_layout(
        renderer_fig,
        renderer_legend,
        layout=renderer_layout,
    )
    plt.close(renderer_fig)

    single_column_fig, single_column_ax = plt.subplots()
    for index in range(10):
        single_column_ax.plot([2030, 2031], [index, index + 1], label=f"line {index}")
    assert footer_mod.render_below_figure_legend(single_column_fig, max_columns=1) is True
    assert [text.get_text() for text in single_column_fig.legends[0].texts] == [
        f"line {index}" for index in range(10)
    ]
    plt.close(single_column_fig)
