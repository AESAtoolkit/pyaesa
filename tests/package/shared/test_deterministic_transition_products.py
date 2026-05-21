from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from pyaesa import set_workspace
from pyaesa.shared.figures import deterministic_transition_products as products_mod
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN


def _title_parts() -> dict[str, str | None]:
    return {
        "family": "Demo family",
        "selector_scope": "r_p=FR",
        "lcia_method": None,
        "user_facing_override_label": None,
        "prospective_scope": None,
    }


def _reuse_variant_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031],
            "value": [1.0, 2.0, 3.0, 4.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP1", "SSP1", "SSP1"],
            "l1_l2_method": ["A", "A", "A", "A"],
            "l2_reuse_year": [2030, 2030, 2031, 2031],
        }
    )


def _multi_panel_variant_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031, 2030, 2031, 2030, 2031],
            "value": [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"] * 8,
            "l1_l2_method": ["A"] * 8,
            "impact": [
                "climate",
                "climate",
                "climate",
                "climate",
                "land",
                "land",
                "land",
                "land",
            ],
            "l2_reuse_year": [2030, 2030, 2031, 2031, None, None, None, None],
            "reference_year": [None, None, None, None, 2020, 2020, 2030, 2030],
            "series_label": ["A"] * 8,
        }
    )


def _plain_transition_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031, 2030, 2031],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"] * 6,
            "series_label": [
                "Series A",
                "Series A",
                "Series B",
                "Series B",
                "Series C",
                "Series C",
            ],
            "impact": ["AAL", "AAL", "SOD", "SOD", "OA", "OA"],
            "lcia_method": ["pb_lcia"] * 6,
        }
    )


def _five_panel_frame() -> pd.DataFrame:
    impacts = ["AAL", "SOD", "OA", "N", "P GLO"]
    rows: list[dict[str, object]] = []
    for index, impact in enumerate(impacts, start=1):
        for year, value in [(2030, float(index)), (2031, float(index + 1))]:
            rows.append(
                {
                    "year": year,
                    "value": value,
                    ASOCC_SSP_SCENARIO_COLUMN: "SSP1",
                    "series_label": f"Series {impact}",
                    "impact": impact,
                    "lcia_method": "pb_lcia",
                }
            )
    return pd.DataFrame(rows)


def test_transition_product_contracts_cover_notes_styles_and_split_lines() -> None:
    frame = _reuse_variant_frame()

    entries, combined_main, combined_note, single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=frame,
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )
    assert len(entries) == 1
    assert not combined_main.empty
    assert combined_note is not None
    assert single_year_note is not None

    specs = products_mod._build_variant_line_specs(  # noqa: SLF001
        entries,
        default_colors=["#111111", "#222222"],
    )
    assert specs
    assert any(line_spec.line_style == "dotted" for _color, line_spec in specs.values())
    assert any(line_spec.prospective_only for _color, line_spec in specs.values())
    impact_specs = products_mod._build_variant_line_specs(  # noqa: SLF001
        [
            (
                "climate",
                frame.assign(impact="climate", series_label="A climate"),
                entries[0][2],
                entries[0][3],
            )
        ],
        default_colors=["#111111", "#222222"],
        include_panel_column_in_series=True,
    )
    assert impact_specs

    fallback = products_mod._fallback_series_colors(  # noqa: SLF001
        n_colors=3,
        excluded={"#006ba4"},
    )
    assert len(fallback) == 3
    assert "#006ba4" not in fallback

    fig, axis = plt.subplots()
    products_mod._plot_series(  # noqa: SLF001
        axis,
        years=[2030, 2031, 2032],
        values=[1.0, 2.0, 3.0],
        label="Visible",
        legend_group="One step methods",
        color="#333333",
        line_style="dotted",
        show_in_legend=True,
        prospective_only=True,
        transition_year=2031,
    )
    assert len(axis.lines) == 2
    assert axis.lines[-1].get_label() == "Visible"
    plt.close(fig)
    fig, axis = plt.subplots()
    products_mod._format_year_axis(axis, requested_years=list(range(2000, 2022)))  # noqa: SLF001
    assert isinstance(axis.xaxis.get_major_locator(), MaxNLocator)
    plt.close(fig)


def test_transition_product_contracts_cover_empty_specs_and_additional_plot_branches() -> None:
    assert (
        products_mod._build_variant_line_specs(  # noqa: SLF001
            [],
            default_colors=["#111111"],
        )
        == {}
    )
    assert (
        products_mod._fallback_series_colors(  # noqa: SLF001
            n_colors=0,
            excluded=set(),
        )
        == []
    )

    fig, axis = plt.subplots()
    products_mod._plot_series(  # noqa: SLF001
        axis,
        years=[2030, 2031],
        values=[1.0, 2.0],
        label="History only",
        legend_group="Methods",
        color=None,
        line_style="dashed",
        show_in_legend=False,
        prospective_only=True,
        transition_year=2035,
    )
    products_mod._plot_series(  # noqa: SLF001
        axis,
        years=[2030, 2031],
        values=[3.0, 4.0],
        label="Prospective only",
        legend_group="Methods",
        color="#444444",
        line_style="dotted",
        show_in_legend=True,
        prospective_only=True,
        transition_year=2030,
    )
    assert len(axis.lines) == 2
    assert axis.lines[0].get_label() == "_nolegend_"
    assert axis.lines[1].get_label() == "Prospective only"
    plt.close(fig)

    first_fallback = products_mod._fallback_series_colors(  # noqa: SLF001
        n_colors=1,
        excluded=set(),
    )[0]
    skipped = products_mod._fallback_series_colors(  # noqa: SLF001
        n_colors=2,
        excluded={first_fallback},
    )
    assert len(skipped) == 2
    assert first_fallback not in skipped

    large_variant_rows: list[dict[str, object]] = []
    for index in range(21):
        l1_l2_method = f"Method {index}"
        for reference_year, base_value in [(2020, 1.0), (2030, 2.0)]:
            for year in [2030, 2031]:
                large_variant_rows.append(
                    {
                        "year": year,
                        "value": base_value + year - 2030,
                        ASOCC_SSP_SCENARIO_COLUMN: "SSP1",
                        "l1_l2_method": l1_l2_method,
                        "series_label": f"{l1_l2_method} {reference_year}",
                        "reference_year": reference_year,
                    }
                )
    large_entries, _combined, _note, _single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=pd.DataFrame(large_variant_rows),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )
    expanded_specs = products_mod._build_variant_line_specs(  # noqa: SLF001
        large_entries,
        default_colors=["#111111"],
    )
    assert len(expanded_specs) == 42

    hsv_map = products_mod.colormaps["hsv"]
    fully_excluded = {
        str(products_mod.to_hex(hsv_map(index / 24.0), keep_alpha=False)) for index in range(24)
    }
    assert (
        products_mod._fallback_series_colors(  # noqa: SLF001
            n_colors=1,
            excluded=fully_excluded,
        )
        == []
    )


def test_panel_variant_entries_join_panel_specific_notes() -> None:
    entries, combined_main, combined_note, single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=_multi_panel_variant_frame(),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )

    assert len(entries) == 2
    assert set(combined_main["impact"].tolist()) == {"climate", "land"}
    assert combined_note is not None
    assert single_year_note is not None


def test_render_multi_year_panels_routes_overlay_split_grid_and_empty_groups(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    hook_calls: list[str] = []
    label_calls: list[str] = []
    transformed_markers: list[int] = []

    def axis_styler(axis, frame: pd.DataFrame) -> None:
        hook_calls.append(str(frame["impact"].iloc[0]))
        axis.axhline(0.0, color="#cccccc", linewidth=0.5)

    def marker_transform(frame: pd.DataFrame, markers: list) -> list:
        transformed_markers.append(len(markers))
        return list(markers)

    def ylabel_resolver(frame: pd.DataFrame) -> str:
        label = f"Value::{frame['impact'].iloc[0]}"
        label_calls.append(label)
        return label

    groups, _combined, footer_note, _single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=_plain_transition_frame(),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )

    overlay_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=groups[:1],
        requested_years=[2030, 2031],
        output_base=tmp_path / "overlay",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=True,
        axis_styler=axis_styler,
        marker_transform=marker_transform,
        ylabel_resolver=None,
        split_panels=False,
        overlay_panels=True,
        footer_note=footer_note,
        force_zero_ymin=True,
    )
    assert overlay_paths == [tmp_path / "overlay.png"]
    assert overlay_paths[0].is_file()

    split_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=groups[:2],
        requested_years=[2030, 2031],
        output_base=tmp_path / "split",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=True,
        axis_styler=axis_styler,
        marker_transform=marker_transform,
        ylabel_resolver=ylabel_resolver,
        split_panels=True,
        overlay_panels=False,
        footer_note=footer_note,
        force_zero_ymin=True,
    )
    assert {path.name for path in split_paths} == {"split__AAL.png", "split__OA.png"}
    assert all(path.is_file() for path in split_paths)

    grid_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=groups,
        requested_years=[2030, 2031],
        output_base=tmp_path / "grid",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=True,
        axis_styler=axis_styler,
        marker_transform=marker_transform,
        ylabel_resolver=ylabel_resolver,
        split_panels=False,
        overlay_panels=False,
        footer_note=footer_note,
        force_zero_ymin=True,
    )
    assert grid_paths == [tmp_path / "grid.png"]
    assert grid_paths[0].is_file()

    empty_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=[],
        requested_years=[2030, 2031],
        output_base=tmp_path / "empty",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=False,
        overlay_panels=False,
        footer_note=None,
        force_zero_ymin=False,
    )
    assert empty_paths == [tmp_path / "empty.png"]
    assert empty_paths[0].is_file()

    assert hook_calls == ["AAL", "AAL", "OA", "AAL", "OA", "SOD"]
    assert label_calls == [
        "Value::AAL",
        "Value::OA",
        "Value::AAL",
        "Value::OA",
        "Value::SOD",
    ]
    assert transformed_markers == [0, 0, 0, 0, 0, 0]


def test_render_multi_year_panels_covers_variant_specs_marker_defaults_and_hidden_axes(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    scenario_values = [None, "SSP1", None, "SSP1", None, "SSP1", None, "SSP1"]
    variant_groups, _combined, footer_note, _single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=_multi_panel_variant_frame().assign(
            **{ASOCC_SSP_SCENARIO_COLUMN: scenario_values},
            lcia_method=["pb_lcia"] * 8,
            impact=["AAL", "AAL", "AAL", "AAL", "SOD", "SOD", "SOD", "SOD"],
            series_label=["A min", "A min", "A max", "A max", "B min", "B min", "B max", "B max"],
        ),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )
    overlay_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=variant_groups,
        requested_years=[2030, 2031],
        output_base=tmp_path / "variant_overlay",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=False,
        overlay_panels=True,
        footer_note=footer_note,
        force_zero_ymin=False,
    )
    assert overlay_paths == [tmp_path / "variant_overlay.png"]
    assert overlay_paths[0].is_file()

    split_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=variant_groups,
        requested_years=[2030, 2031],
        output_base=tmp_path / "variant_split",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=True,
        overlay_panels=False,
        footer_note=footer_note,
        force_zero_ymin=False,
    )
    assert {path.name for path in split_paths} == {
        "variant_split__AAL.png",
        "variant_split__SOD.png",
    }

    grid_groups, _combined, _footer_note, _single_year_note = products_mod._panel_variant_entries(  # noqa: SLF001
        frame=_five_panel_frame(),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#123456",
    )
    grid_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=grid_groups,
        requested_years=[2030, 2031],
        output_base=tmp_path / "five_panel_grid",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=False,
        overlay_panels=False,
        footer_note=None,
        force_zero_ymin=False,
    )
    assert grid_paths == [tmp_path / "five_panel_grid.png"]
    assert grid_paths[0].is_file()

    single_panel_paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=variant_groups[:1],
        requested_years=[2030, 2031],
        output_base=tmp_path / "single_panel_grid",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=False,
        overlay_panels=False,
        footer_note=None,
        force_zero_ymin=False,
    )
    assert single_panel_paths == [tmp_path / "single_panel_grid.png"]
    assert single_panel_paths[0].is_file()


def test_render_multi_year_split_skips_legend_for_empty_payloads(tmp_path: Path) -> None:
    set_workspace(tmp_path)
    empty_group = (
        "AAL",
        pd.DataFrame(columns=["year", "value", "impact", "lcia_method", "series_label"]),
        tuple(),
        None,
    )
    paths = products_mod._render_multi_year_panels(  # noqa: SLF001
        groups=[empty_group],
        requested_years=[2030, 2031],
        output_base=tmp_path / "empty_split",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        value_scale=1.0,
        percent_ticks=False,
        axis_styler=None,
        marker_transform=None,
        ylabel_resolver=None,
        split_panels=True,
        overlay_panels=False,
        footer_note=None,
        force_zero_ymin=False,
    )
    assert paths == [tmp_path / "empty_split__AAL.png"]
    assert paths[0].is_file()


def test_render_transition_products_routes_multi_year_and_checkpoint_products(
    tmp_path: Path,
) -> None:
    checkpoint_calls: list[tuple[list[int], Path]] = []

    def single_year_renderer(**kwargs) -> list[Path]:
        output_base = kwargs["output_base"]
        checkpoint_calls.append((list(kwargs["years"]), output_base))
        path = output_base.parent / f"{output_base.name}__checkpoint.png"
        path.write_text("checkpoint", encoding="utf-8")
        return [path]

    long_frame = pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031],
            "value": [1.0, 2.0, 3.0, 4.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP1", "SSP1", "SSP1"],
            "series_label": ["Series A", "Series A", "Series B", "Series B"],
            "l1_l2_method": ["A", "A", "B", "B"],
        }
    )

    paths = products_mod.render_transition_products(
        long_frame=long_frame,
        requested_years=[2030, 2031],
        output_base=tmp_path / "transition",
        family="Demo family",
        ylabel="Value",
        dpi=10,
        output_format="png",
        marker_label="switch",
        marker_color="#123456",
        single_year_renderer=single_year_renderer,
        include_single_year_products=True,
        force_zero_ymin=True,
    )

    assert len(paths) == 2
    assert any(path.name == "transition__prospective_SSP1.png" for path in paths)
    assert any(path.name == "transition__prospective_SSP1__checkpoint.png" for path in paths)
    assert all(path.is_file() for path in paths)
    assert len(checkpoint_calls) == 1
    assert checkpoint_calls[0][1] == tmp_path / "transition__prospective_SSP1"


def test_render_transition_products_with_series_labels_uses_user_facing_override_label_columns(
    tmp_path: Path,
) -> None:
    long_frame = pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031],
            "value": [10.0, 20.0, 30.0, 40.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP1", "SSP1", "SSP1"],
            "l1_l2_method": ["Method A", "Method A", "Method B", "Method B"],
        }
    )

    paths = products_mod.render_transition_products_with_series_labels(
        long_frame=long_frame,
        label_columns=("l1_l2_method",),
        skip_columns=None,
        context="Demo transition renderer",
        requested_years=[2030, 2031],
        output_base=tmp_path / "series_labels",
        family="Demo family",
        ylabel="Value",
        dpi=10,
        output_format="png",
        marker_label="switch",
        marker_color="#123456",
        include_single_year_products=False,
        force_zero_ymin=True,
    )

    assert paths == [tmp_path / "series_labels__prospective_SSP1.png"]
    assert paths[0].is_file()


def test_render_transition_products_with_series_labels_repeats_history_into_dynamic_model_scope(
    tmp_path: Path,
) -> None:
    long_frame = pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "value": [10.0, 20.0, 30.0],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None, "SSP2"],
            "cc_model": [None, None, "Model A"],
            "cc_scenario": [None, None, "Scenario A"],
            "l1_l2_method": ["Method A", "Method A", "Method A"],
        }
    )

    paths = products_mod.render_transition_products_with_series_labels(
        long_frame=long_frame,
        label_columns=("l1_l2_method",),
        skip_columns=None,
        context="Dynamic transition renderer",
        requested_years=[2020, 2021, 2022],
        output_base=tmp_path / "dynamic_series_labels",
        family="Demo family",
        ylabel="Value",
        dpi=10,
        output_format="png",
        marker_label="switch",
        marker_color="#123456",
        include_single_year_products=False,
        force_zero_ymin=True,
        transition_grouping_skip_columns={"cc_model", "cc_scenario"},
    )

    assert paths == [tmp_path / "dynamic_series_labels__prospective_SSP2.png"]
    assert paths[0].is_file()


def test_render_transition_products_exact_single_year_scope_skips_multi_year_render(
    tmp_path: Path,
) -> None:
    checkpoint_calls: list[tuple[list[int], Path]] = []

    def single_year_renderer(**kwargs) -> list[Path]:
        output_base = kwargs["output_base"]
        checkpoint_calls.append((list(kwargs["years"]), output_base))
        path = output_base.parent / f"{output_base.name}__checkpoint.png"
        path.write_text("checkpoint", encoding="utf-8")
        return [path]

    paths = products_mod.render_transition_products(
        long_frame=pd.DataFrame(
            {
                "year": [2030],
                "value": [1.0],
                "series_label": ["Series A"],
                "l1_l2_method": ["A"],
            }
        ),
        requested_years=[2030],
        output_base=tmp_path / "single_year_only",
        family="Demo family",
        ylabel="Value",
        dpi=10,
        output_format="png",
        marker_label="switch",
        marker_color="#123456",
        single_year_renderer=single_year_renderer,
        include_single_year_products=False,
    )

    assert paths == [tmp_path / "single_year_only__checkpoint.png"]
    assert checkpoint_calls == [([2030], tmp_path / "single_year_only")]


def test_default_scoped_output_base_suffixes_selector_scenario_and_impact_tokens(
    tmp_path: Path,
) -> None:
    scoped = products_mod._default_scoped_output_base(
        output_base=tmp_path / "base",
        lcia_method="gwp100_lcia",
        selector_token="r_p_FR",
        scenario_token="prospective_SSP2",
        impact_token="impact_GWP_100",
    )

    assert scoped.name == "base__gwp100_lcia__r_p_FR__prospective_SSP2__impact_GWP_100"


def test_render_transition_products_groups_combined_impacts_by_lcia_selector_and_scenario(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    long_frame = pd.DataFrame(
        {
            "year": [2030, 2031] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"] * 8,
            "series_label": [
                "Series A",
                "Series A",
                "Series A",
                "Series A",
                "Series B",
                "Series B",
                "Series B",
                "Series B",
            ],
            "impact": ["AAL", "AAL", "SOD", "SOD", "AAL", "AAL", "SOD", "SOD"],
            "lcia_method": ["pb_lcia"] * 8,
            "r_p": ["FR", "FR", "FR", "FR", "DE", "DE", "DE", "DE"],
        }
    )

    paths = products_mod.render_transition_products(
        long_frame=long_frame,
        requested_years=[2030, 2031],
        output_base=tmp_path / "combined",
        family="Demo family",
        ylabel="Value",
        dpi=10,
        output_format="png",
        marker_label="switch",
        marker_color="#123456",
        include_single_year_products=False,
        group_combined_by_impact=True,
        selector_scope_request=products_mod.SelectorScopeRequest(axes=(("r_p", ("FR", "DE")),)),
    )

    assert {path.name for path in paths} == {
        "combined__pb_lcia__rp_DE__prospective_SSP1__AAL.png",
        "combined__pb_lcia__rp_DE__prospective_SSP1__SOD.png",
        "combined__pb_lcia__rp_FR__prospective_SSP1__AAL.png",
        "combined__pb_lcia__rp_FR__prospective_SSP1__SOD.png",
    }
    assert all(path.is_file() for path in paths)
