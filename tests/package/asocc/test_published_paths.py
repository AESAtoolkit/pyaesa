from pathlib import Path
from types import SimpleNamespace

from pyaesa.asocc.runtime.paths.family_roots import _get_asocc_root, asocc_source_version_token
from pyaesa.asocc.runtime.paths import published as paths_mod


def _common(tmp_path):
    return {
        "proj_base": tmp_path,
        "source": "oecd_v2025",
        "agg_version": None,
    }


def _scope_root_args(common: dict) -> dict:
    return {
        "proj_base": common["proj_base"],
        "source": common["source"],
        "agg_version": common["agg_version"],
    }


def test_published_path_cover_scope_and_level_validation(tmp_path) -> None:
    common = _common(tmp_path)
    family_root = _get_asocc_root(proj_base=tmp_path)
    source_token = asocc_source_version_token(
        source=common["source"],
        agg_version=common["agg_version"],
    )
    assert paths_mod._owning_fu_level_for_code(fu_code=None) == "level_1"  # noqa: SLF001
    assert paths_mod._owning_fu_level_for_code(fu_code="L2.a.a") == "level_2"  # noqa: SLF001
    assert paths_mod._normalize_owning_fu_level(owning_fu_level=" level_1 ") == "level_1"  # noqa: SLF001
    scope_root = paths_mod._asocc_deterministic_scope_root(**_scope_root_args(common))  # noqa: SLF001
    assert scope_root == family_root / source_token / "deterministic"
    assert paths_mod._get_asocc_results_root(**common) == scope_root / "results"  # noqa: SLF001
    assert (
        paths_mod._get_asocc_l2_results_root(  # noqa: SLF001
            bucket="l2_vs_global",
            **common,
        )
        == scope_root / "results" / "level_2" / "l2_vs_global"
    )
    assert (
        paths_mod._canonical_l2_results_relative_dir(  # noqa: SLF001
            bucket="l2_in_l1",
        )
        == Path("results") / "level_2" / "l2_in_l1"
    )
    assert paths_mod._get_asocc_figures_root(level="level_1", **common) == (  # noqa: SLF001
        scope_root / "figures"
    )
    assert (
        paths_mod._canonical_l2_results_relative_dir(bucket="")  # noqa: SLF001
        == Path("results") / "level_2" / "l2_vs_global"
    )

    disagg_scope_root = paths_mod._asocc_deterministic_scope_root(  # noqa: SLF001
        proj_base=tmp_path,
        source="disagg_oecd",
        agg_version="elec",
    )
    assert disagg_scope_root == family_root / "disagg_oecd" / "deterministic"


def test_share_and_enacting_metric_path_cover_suffix_and_stem_contracts(tmp_path) -> None:
    common = _common(tmp_path)
    deterministic_root = paths_mod._asocc_deterministic_scope_root(**_scope_root_args(common))  # noqa: SLF001

    assert paths_mod._get_asocc_l1_dir(lcia_sub=None, **common) == (  # noqa: SLF001
        deterministic_root / "results"
    )
    assert paths_mod._get_asocc_l1_dir(  # noqa: SLF001
        lcia_sub=None,
        owning_fu_level="level_2",
        **common,
    ) == (deterministic_root / "results" / "level_1")
    assert paths_mod._get_asocc_l2_dir(  # noqa: SLF001
        bucket="l2_vs_global",
        lcia_sub="regression_proj",
        **common,
    ) == (deterministic_root / "results" / "level_2" / "l2_vs_global" / "regression_proj")
    assert paths_mod._get_asocc_l2_dir(  # noqa: SLF001
        bucket="l2_in_l1",
        lcia_sub=None,
        **common,
    ) == (deterministic_root / "results" / "level_2" / "l2_in_l1")

    assert paths_mod._get_enacting_metric_dir(  # noqa: SLF001
        level="level_2",
        lcia_sub="regression_proj",
        **common,
    ) == (deterministic_root / "results" / "level_2" / "enacting_metrics" / "regression_proj")

    assert paths_mod._get_enacting_metric_output_path(  # noqa: SLF001
        level="level_1",
        key_metric="e_pba_reg_capita",
        key_method="gwp100_lcia",
        key_scenario="SSP2",
        output_format="csv",
        lcia_sub=None,
        **common,
    ) == (
        deterministic_root / "results" / "enacting_metrics" / "e_pba_reg_cap_SSP2_gwp100_lcia.csv"
    )
    assert paths_mod._get_enacting_metric_output_path(  # noqa: SLF001
        level="level_1",
        key_metric="e_pba_reg_capita",
        key_method="gwp100_lcia",
        key_scenario="SSP2",
        output_format="csv",
        lcia_sub=None,
        owning_fu_level="level_2",
        **common,
    ) == (
        deterministic_root
        / "results"
        / "level_1"
        / "enacting_metrics"
        / "e_pba_reg_cap_SSP2_gwp100_lcia.csv"
    )
    assert paths_mod._get_enacting_metric_output_path(  # noqa: SLF001
        level="level_2",
        key_metric="population",
        key_method=None,
        key_scenario=None,
        output_format="pickle",
        lcia_sub="regression_proj",
        **common,
    ) == (
        deterministic_root
        / "results"
        / "level_2"
        / "enacting_metrics"
        / "regression_proj"
        / "population.pickle"
    )


def test_figure_and_reuse_paths_cover_deterministic_root_contract(tmp_path) -> None:
    common = _common(tmp_path)
    deterministic_root = paths_mod._asocc_deterministic_scope_root(**_scope_root_args(common))  # noqa: SLF001
    assert paths_mod._get_asocc_figures_root(level="level_2", **common) == (  # noqa: SLF001
        deterministic_root / "figures_l2_vs_global"
    )

    context = SimpleNamespace(output_format="parquet", **common)
    assert paths_mod.reuse_output_path_for(
        context=context,
        bucket="l2_vs_global",
        file_stem="demo_output",
    ) == (deterministic_root / "results" / "level_2" / "l2_vs_global" / "demo_output.parquet")
