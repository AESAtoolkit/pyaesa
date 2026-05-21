from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from pyaesa.shared.figures import deterministic_legends as legends_mod
from pyaesa.shared.figures import deterministic_legends_methods as methods_mod


def test_grouped_legend_rendering_and_ordering() -> None:
    fig, ax = plt.subplots()
    consumption_line = ax.plot([2030, 2031], [1.0, 2.0], label="Consumption")[0]
    one_step_line = ax.plot([2030, 2031], [2.0, 3.0], label="One step")[0]
    legends_mod.bind_deterministic_legend_group(
        consumption_line,
        "Consumption-anchored two steps",
    )
    legends_mod.bind_deterministic_legend_group(one_step_line, "One-step")

    legends_mod.render_grouped_deterministic_legend_below(ax, legend_note="note")

    assert len(fig.legends) == 3
    assert [legend.get_title().get_text() for legend in fig.legends] == [
        "",
        "Consumption-anchored two steps",
        "One-step",
    ]
    assert [text.get_text() for text in fig.legends[0].texts] == ["note"]

    ordered = legends_mod._ordered_groups(  # noqa: SLF001
        OrderedDict(
            {
                "Other group": [(one_step_line, "One step")],
                "Consumption-anchored two steps": [(consumption_line, "Consumption")],
            }
        )
    )
    assert list(ordered) == ["Consumption-anchored two steps", "Other group"]

    plt.close(fig)


def test_grouped_legend_extra_height_applies_to_selected_group() -> None:
    fig, ax = plt.subplots()
    one_step_line = ax.plot([2030, 2031], [1.0, 2.0], label="One step")[0]
    two_step_line = ax.plot([2030, 2031], [2.0, 3.0], label="Two step")[0]
    legends_mod.bind_deterministic_legend_group(one_step_line, "One-step")
    legends_mod.bind_deterministic_legend_group(two_step_line, "Two-step")

    legends_mod.render_grouped_deterministic_legend_below(
        ax,
        legend_kwargs={"handlelength": 1.5},
        legend_kwargs_group_title="One-step",
        legend_extra_height_in=0.2,
    )

    assert [legend.get_title().get_text() for legend in fig.legends] == ["Two-step", "One-step"]
    plt.close(fig)


def test_grouped_legend_single_group_and_duplicate_deduplication() -> None:
    empty_fig, empty_ax = plt.subplots()
    legends_mod.render_grouped_deterministic_legend_below(empty_ax)
    assert len(empty_fig.legends) == 0
    plt.close(empty_fig)

    single_fig, single_ax = plt.subplots()
    line = single_ax.plot([2030, 2031], [1.0, 2.0], label="Visible")[0]
    legends_mod.bind_deterministic_legend_group(line, "One-step")

    legends_mod.render_grouped_deterministic_legend_below(single_ax)

    assert len(single_fig.legends) == 1
    assert single_fig.legends[0].get_title().get_text() == "One-step"
    plt.close(single_fig)

    duplicate_fig, duplicate_ax = plt.subplots()
    first = duplicate_ax.plot([2030, 2031], [1.0, 2.0], label="Same")[0]
    second = duplicate_ax.plot([2030, 2031], [2.0, 3.0], label="Same")[0]
    legends_mod.bind_deterministic_legend_group(first, "One-step")
    legends_mod.bind_deterministic_legend_group(second, "Two-step")

    legends_mod.render_grouped_deterministic_legend_below(duplicate_ax)

    assert len(duplicate_fig.legends) == 1
    assert [text.get_text() for text in duplicate_fig.legends[0].texts] == ["Same"]

    plt.close(duplicate_fig)

    unclassified_fig, unclassified_ax = plt.subplots()
    unclassified_line = unclassified_ax.plot(
        [2030, 2031],
        [1.0, 2.0],
        label="Method\nplain min: ref_year=2020\ndotted max: ref_year=2030",
    )[0]
    legends_mod.bind_deterministic_legend_group(unclassified_line, "")

    legends_mod.render_grouped_deterministic_legend_below(unclassified_ax)

    assert len(unclassified_fig.legends) == 1
    assert unclassified_fig.legends[0].get_title().get_text() == ""

    plt.close(unclassified_fig)

    mixed_fig, mixed_ax = plt.subplots()
    mixed_unclassified = mixed_ax.plot([2030, 2031], [1.0, 2.0], label="L1")[0]
    mixed_one_step = mixed_ax.plot([2030, 2031], [2.0, 3.0], label="L2")[0]
    legends_mod.bind_deterministic_legend_group(mixed_unclassified, "")
    legends_mod.bind_deterministic_legend_group(mixed_one_step, "One-step")

    legends_mod.render_grouped_deterministic_legend_below(mixed_ax)

    assert [legend.get_title().get_text() for legend in mixed_fig.legends] == [
        "One-step",
        "",
    ]

    plt.close(mixed_fig)


def test_legend_group_from_row_classifies_visible_method_families() -> None:
    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_1",
                    "fu_code": "L1.demo",
                    "l1_l2_method": "EG(Pop)",
                }
            )
        )
        == ""
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.a",
                    "l1_l2_method": "EG(Pop)",
                }
            )
        )
        == "One-step"
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.a",
                    "l1_method": "EG(Pop)",
                    "l2_method": "AR(E^{CBA_FD})",
                    "l1_l2_method": "EG(Pop)+AR(E^{CBA_FD})",
                }
            )
        )
        == "Two-step"
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.a",
                    "l2_method": "EG(Pop)",
                    "l1_l2_method": "AR(E^{CBA_FD})",
                }
            )
        )
        == "Two-step"
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.b",
                    "l1_method": "UT(FDa)",
                    "l2_method": "UT(FDa)",
                    "l1_l2_method": "UT(FDa)",
                }
            )
        )
        == "Consumption-anchored two steps"
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.b",
                    "l1_method": "UT(GVAa)",
                    "l2_method": "UT(GVAa)",
                    "l1_l2_method": "UT(GVAa)",
                }
            )
        )
        == "Production-anchored two steps"
    )

    assert (
        methods_mod.legend_group_from_row(
            pd.Series(
                {
                    "level": "level_2",
                    "fu_code": "L2.demo.b",
                    "l1_method": "bad",
                    "l2_method": "bad",
                    "l1_l2_method": "bad",
                }
            )
        )
        == "Two-step"
    )


def test_legend_method_contracts_cover_tokens_and_conflicts() -> None:
    token_row = pd.Series(
        {
            "fu_code": "L2.demo.b",
            "l1_l2_method": "AR(E^{CBA_FD}) + EG(Pop) + AR(E^{CBA_FD})",
        }
    )
    assert methods_mod._row_method_tokens(token_row) == (  # noqa: SLF001
        "AR(E^{CBA_FD})",
        "EG(Pop)",
    )
    assert methods_mod._path_from_label("UT(FDa)") == "consumption"  # noqa: SLF001
    assert methods_mod._path_from_label("UT(GVAa)") == "production"  # noqa: SLF001
    assert methods_mod._path_from_label("bad") is None  # noqa: SLF001
    assert methods_mod._path_from_label(None) is None  # noqa: SLF001
    assert methods_mod._path_from_label("   ") is None  # noqa: SLF001
    assert methods_mod._path_from_tokens(("UT(GVAa)", "EG(Pop)")) == "production"  # noqa: SLF001
    assert methods_mod._is_known_method_token("PR-HR(Ecap,cum)") is True  # noqa: SLF001
    assert methods_mod._is_known_method_token("demo") is False  # noqa: SLF001
    assert methods_mod._is_l1_level(pd.Series({"fu_code": "L1.demo"})) is True  # noqa: SLF001
    assert methods_mod._is_l2_b_fu(pd.Series({"fu_code": "L2.demo.b"})) is True  # noqa: SLF001
    assert methods_mod._value(pd.Series({"demo": " 2.0 "}), "demo") == "2"  # noqa: SLF001
    assert methods_mod._merge_paths(paths=(None, None)) is None  # noqa: SLF001
