from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa import deterministic_asocc
from pyaesa.asocc.figures.file_stems import asocc_scope_stem
from pyaesa.asocc.figures.per_method_renderer import scope_title as asocc_scope_title
from pyaesa.asocc.figures.row_reader import normalize_ssp_rows
from pyaesa.asocc.figures.scope_planner import requested_ssp_scenarios
from pyaesa.asocc.runtime.paths.external import external_asocc_relative_dir
from pyaesa.asocc.uncertainty.figures.scope_planner import scope_title as uncertainty_scope_title
from pyaesa.asocc.runtime.paths.external import get_asocc_external_method_level_dir
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from tests.package.helpers.acc_dummy_repo import prepare_exiobase_repo_with_years


def _run_deterministic_asocc(*, project_name: str, refresh: bool, figures: bool = False):
    return deterministic_asocc(
        project_name=project_name,
        source="oecd_v2025",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=figures,
        figure_format={"format": "svg", "dpi": 1},
        refresh=refresh,
    )


def _asocc_root(repo_root: Path, *, project_name: str) -> Path:
    return repo_root / f"{project_name}" / "B1_asocc"


def test_deterministic_asocc_figure_scope_helpers_cover_selectors_and_empty_ssps(
    read_only_project_repo: Path,
) -> None:
    del read_only_project_repo
    frame = pd.DataFrame(
        {
            "lcia_method": ["gwp100_lcia"],
            "impact": ["GWP_100"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
        }
    )

    assert (
        asocc_scope_stem(
            "multi_method",
            frame,
            include_impact=True,
            selector_token="rp_FR__sp_D",
            studied_year=2030,
        )
        == "multi_method__rp_FR__sp_D__gwp100_lcia__GWP_100__SSP2__2030"
    )
    assert (
        asocc_scope_stem("multi_method", frame, include_impact=False)
        == "multi_method__gwp100_lcia__SSP2"
    )
    assert "France electricity" in asocc_scope_title(
        "aSoCC deterministic",
        "demo",
        frame,
        selector_title="France electricity",
        studied_year=2030,
    )
    assert "France electricity" not in asocc_scope_title(
        "aSoCC deterministic",
        "demo",
        frame,
        selector_title=" ",
    )
    assert "France electricity" not in uncertainty_scope_title(
        "demo",
        frame,
        selector_title=" ",
        studied_year=None,
    )
    assert "GWP_100" in asocc_scope_title(
        "aSoCC deterministic",
        "demo",
        frame,
        selector_title="France electricity",
        studied_year=2030,
    )
    assert "gwp100_lcia" not in asocc_scope_title(
        "aSoCC deterministic",
        "demo",
        frame,
        selector_title="France electricity",
        studied_year=2030,
    )
    assert "GWP_100" in uncertainty_scope_title(
        "demo",
        frame,
        selector_title="France electricity",
        studied_year=None,
    )
    assert "gwp100_lcia" not in uncertainty_scope_title(
        "demo",
        frame,
        selector_title="France electricity",
        studied_year=None,
    )
    assert normalize_ssp_rows(frame, ssp_scenarios=[]).equals(frame)
    assert requested_ssp_scenarios(
        options_by_year=None,
        compute_signature={"ssp_scenario_input": ["ssp2"]},
    ) == ["SSP2"]
    assert requested_ssp_scenarios(
        options_by_year={2030: [None, "SSP2"]},
        compute_signature={},
    ) == [None, "SSP2"]
    assert external_asocc_relative_dir(level="level_1").as_posix() == "results"


def test_deterministic_asocc_end_to_end_reuse_and_refresh(allocation_dummy_repo) -> None:
    report = _run_deterministic_asocc(project_name="asocc_public", refresh=True)

    assert report is not None
    assert report.source == "oecd_v2025"
    assert len(report.summaries) == 1

    output_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_public",
        ).rglob("*.csv")
    )
    result_paths = [path for path in output_paths if path.name == "UT(FD).csv"]
    assert len(result_paths) == 1
    output_frame = pd.read_csv(result_paths[0])
    assert {"r_p", "s_p", "2005"}.issubset(output_frame.columns)
    assert set(output_frame["r_p"]) == {"FR", "US"}
    assert set(output_frame["s_p"]) == {"D", "X"}
    assert bool(output_frame["2005"].gt(0).all())

    external_root = allocation_dummy_repo.repo_root / "asocc_public" / "B1_asocc"
    assert (external_root / "external_asocc" / "deterministic" / "CO(S).csv").exists()
    assert (external_root / "external_asocc" / "deterministic" / "CO(S)__ssp2.csv").exists()
    assert (
        allocation_dummy_repo.repo_root
        / "asocc_public"
        / "A_lca"
        / "external_lca"
        / "deterministic"
        / "template__ef_3.1.csv"
    ).exists()
    assert not any(external_root.rglob("*.png"))

    reused_report = _run_deterministic_asocc(project_name="asocc_public", refresh=False)
    assert reused_report.reuse_status == "reused_exact"

    refreshed_report = _run_deterministic_asocc(project_name="asocc_public", refresh=True)

    assert refreshed_report is not None


@pytest.mark.parametrize("refresh", [False, True])
def test_deterministic_asocc_rejects_shared_scope_identity_drift(
    allocation_dummy_repo,
    refresh: bool,
) -> None:
    del allocation_dummy_repo
    _run_deterministic_asocc(project_name="asocc_identity_guard", refresh=True)

    def rerun():
        return deterministic_asocc(
            project_name="asocc_identity_guard",
            source="oecd_v2025",
            years=[2005],
            fu_code="L2.a.a",
            r_p=["FR"],
            method_plan="one_step",
            one_step_methods=["UT(FD)"],
            figures=False,
            refresh=refresh,
        )

    if refresh:
        refreshed = rerun()
        assert refreshed is not None
    else:
        with pytest.raises(ValueError):
            rerun()


def test_deterministic_asocc_generates_figures_for_native_method_scope(
    allocation_dummy_repo,
) -> None:
    report = _run_deterministic_asocc(
        project_name="asocc_figures_public",
        refresh=True,
        figures=True,
    )

    assert report is not None
    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public",
        ).rglob("*.svg")
    )
    assert figure_paths
    assert any("UT_FD" in path.as_posix() for path in figure_paths)

    restyled = deterministic_asocc(
        project_name="asocc_figures_public",
        source="oecd_v2025",
        years=[2005, 2006],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=True,
        figure_format={"format": "pdf", "dpi": 10},
        refresh=False,
    )

    assert restyled.figure_paths
    assert all(path.exists() for path in restyled.figure_paths)
    exact_reuse = deterministic_asocc(
        project_name="asocc_figures_public",
        source="oecd_v2025",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=True,
        figure_format={"format": "pdf", "dpi": 10},
        refresh=False,
    )
    assert exact_reuse.reuse_status == "reused_exact"
    no_product = deterministic_asocc(
        project_name="asocc_figures_public",
        source="oecd_v2025",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=True,
        figure_options={"per_method": False, "multi_method": False},
        figure_format={"format": "pdf", "dpi": 10},
        refresh=False,
    )
    assert no_product.figure_paths == []


def test_deterministic_asocc_generates_public_level_1_figures(
    allocation_dummy_repo,
) -> None:
    deterministic_asocc(
        project_name="asocc_figures_public_level_1",
        source="exiobase_396_ixi",
        years=[2005],
        fu_code="L1.a",
        method_plan="default",
        l1_methods=["EG(Pop)"],
        l1_reg_aggreg="pre",
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )

    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_level_1",
        ).rglob("*.svg")
    )

    assert figure_paths
    assert any("EG_Pop" in path.as_posix() for path in figure_paths)


def test_deterministic_asocc_figures_reject_public_scope_without_outputs(
    allocation_dummy_repo,
) -> None:
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="asocc_figures_public_level_1_empty_scope",
            source="exiobase_396_ixi",
            years=[2005],
            fu_code="L1.a",
            method_plan="default",
            l1_methods=["EG(Pop)"],
            r_f=["FR"],
            l1_reg_aggreg="pre",
            figures=True,
            figure_format={"format": "svg", "dpi": 1},
            refresh=True,
        )


def test_deterministic_asocc_figures_include_public_external_method(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_figures_public_external_method"
    base_args = {
        "project_name": project_name,
        "source": "oecd_v2025",
        "years": [2005],
        "fu_code": "L2.a.a",
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
    }
    deterministic_asocc(figures=False, refresh=True, **base_args)
    external_dir = get_asocc_external_method_level_dir(
        proj_base=allocation_dummy_repo.repo_root / f"{project_name}",
        storage_mode="deterministic",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"r_p": ["FR"], "s_p": ["D"], "2005": [0.0]}).to_csv(
        external_dir / "UT(TD).csv",
        index=False,
    )

    deterministic_asocc(
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        figure_external_method={"one_step_methods": ["UT(TD)"]},
        refresh=False,
        **base_args,
    )
    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name=project_name,
        ).rglob("*.svg")
    )

    assert figure_paths
    rendered = "\n".join(path.as_posix() for path in figure_paths)
    assert "UT_TD" in rendered
    assert "multi_method" in rendered


def test_deterministic_asocc_figures_reject_missing_public_external_method(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_figures_public_missing_external_method"
    base_args = {
        "project_name": project_name,
        "source": "oecd_v2025",
        "years": [2005],
        "fu_code": "L2.a.a",
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
    }
    deterministic_asocc(figures=False, refresh=True, **base_args)

    with pytest.raises(ValueError):
        deterministic_asocc(
            figures=True,
            figure_format={"format": "svg", "dpi": 1},
            figure_external_method={"one_step_methods": ["UT(TD)"]},
            refresh=False,
            **base_args,
        )


def test_deterministic_asocc_generates_public_variant_and_transition_figures(
    allocation_dummy_repo,
) -> None:
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2018, 2019, 2020],
        scenario_years=[2020, 2030, 2031],
    )
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=[2018, 2019],
        scenario_years=[2020, 2030, 2031],
    )
    common_args = {
        "source": "exiobase_396_ixi",
        "fu_code": "L2.a.b",
        "method_plan": "pairs",
        "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
        "lcia_method": "gwp100_lcia",
        "r_p": ["FR"],
        "s_p": ["D"],
        "l1_reg_aggreg": "pre",
        "ssp_scenario": ["SSP2"],
        "figures": True,
        "figure_format": {"format": "svg", "dpi": 1},
        "refresh": True,
    }

    deterministic_asocc(
        project_name="asocc_figures_public_transition",
        years=[2018, 2019, 2020],
        reference_years=[2018],
        **common_args,
    )
    transition_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_transition",
        ).rglob("*.svg")
    )

    deterministic_asocc(
        project_name="asocc_figures_public_single_variants",
        years=[2030],
        reference_years=[2018, 2019],
        projection_mode="historical_reuse",
        reg_window=[2018, 2019],
        l2_reuse_years=[2018, 2019],
        **common_args,
    )
    variant_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_single_variants",
        ).rglob("*.svg")
    )

    assert any("multi_method" in path.parts for path in transition_paths)
    assert any("per_method" in path.parts for path in transition_paths)
    assert any("SSP2" in path.stem for path in transition_paths)
    assert any(path.stem.endswith("2030") for path in variant_paths)


def test_deterministic_asocc_generates_public_regression_route_figures(
    allocation_dummy_repo,
) -> None:
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2017, 2018, 2019],
        scenario_years=[2030],
    )
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=[2017, 2018, 2019],
        scenario_years=[2030],
    )

    deterministic_asocc(
        project_name="asocc_figures_public_regression_route",
        source="exiobase_396_ixi",
        years=[2030],
        reference_years=[2017],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        lcia_method="gwp100_lcia",
        r_p=["FR"],
        s_p=["D"],
        l1_reg_aggreg="pre",
        projection_mode="regression",
        reg_window=[2017, 2018, 2019],
        ssp_scenario=["SSP2"],
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )
    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_regression_route",
        ).rglob("*.svg")
    )

    assert figure_paths


def test_deterministic_asocc_generates_public_long_year_axis_figures(
    allocation_dummy_repo,
) -> None:
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=list(range(1995, 2021)),
        scenario_years=[],
    )

    deterministic_asocc(
        project_name="asocc_figures_public_long_year_axis",
        source="exiobase_396_ixi",
        years=list(range(1995, 2021)),
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        r_p=["FR"],
        s_p=["D"],
        l1_reg_aggreg="pre",
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )
    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_long_year_axis",
        ).rglob("*.svg")
    )

    assert figure_paths


def test_deterministic_asocc_generates_public_multi_impact_single_year_figures(
    allocation_dummy_repo,
) -> None:
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="pb_lcia",
        available_years=list(range(1995, 2007)),
        impacts=["aal_child", "bifd_child"],
        impact_parents={"aal_child": "AAL", "bifd_child": "BI FD"},
    )

    deterministic_asocc(
        project_name="asocc_figures_public_multi_impact",
        source="exiobase_396_ixi",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step_pairs",
        one_step_methods=["AR(E^{CBA_FD})"],
        l1_l2_pairs=[
            "AR(E^{CBA_FD})::UT(FD)",
            "EG(Pop)::AR(E^{CBA_FD})",
        ],
        lcia_method="pb_lcia",
        r_p=["FR"],
        s_p=["D"],
        r_f=["FR"],
        l1_reg_aggreg="pre",
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )
    figure_paths = sorted(
        _asocc_root(
            allocation_dummy_repo.repo_root,
            project_name="asocc_figures_public_multi_impact",
        ).rglob("*.svg")
    )

    assert any("multi_method" in path.parts for path in figure_paths)
    assert any(path.stem.endswith("2005") for path in figure_paths)


def test_deterministic_asocc_rejects_figure_external_method_when_figures_disabled(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(
        ValueError,
    ):
        deterministic_asocc(
            project_name="asocc_invalid_figure_external_method",
            source="oecd_v2025",
            years=[2005],
            fu_code="L2.a.a",
            method_plan="one_step",
            one_step_methods=["UT(FD)"],
            figures=False,
            figure_external_method={"one_step_methods": ["AR\\(E\\^\\{CBA_FD\\}\\)"]},
            refresh=True,
        )


def test_deterministic_asocc_rejects_oecd_lcia_method(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="asocc_oecd_lcia_invalid",
            source="oecd_v2025",
            years=[2005],
            fu_code="L1.a",
            method_plan="default",
            l1_methods=["AR(E^{CBA_FD})"],
            lcia_method="gwp100_lcia",
            figures=False,
            refresh=True,
        )


def test_deterministic_asocc_rejects_none_source(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="asocc_missing_source",
            source=cast(Any, None),
            years=[2005],
            fu_code="L2.a.a",
            method_plan="one_step",
            one_step_methods=["UT(FD)"],
            figures=False,
            refresh=True,
        )


def test_deterministic_asocc_rejects_empty_source(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="asocc_empty_source",
            source="",
            years=[2005],
            fu_code="L2.a.a",
            method_plan="one_step",
            one_step_methods=["UT(FD)"],
            figures=False,
            refresh=True,
        )


def test_deterministic_asocc_rejects_non_boolean_group_indices(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="asocc_invalid_group_indices",
            source="oecd_v2025",
            years=[2005],
            fu_code="L2.a.a",
            method_plan="one_step",
            one_step_methods=["UT(FD)"],
            group_indices=cast(Any, "both"),
            figures=False,
            refresh=True,
        )
