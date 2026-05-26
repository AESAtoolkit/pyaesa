from pathlib import Path

import pytest

from pyaesa import set_workspace
from pyaesa.workspace_initialisation.packaged_prerequisites import (
    import_prerequisites,
    packaged_prerequisites_root,
)
from pyaesa.workspace_initialisation.workspace import (
    clear_default_repo_root,
    get_default_repo_root,
    project_outputs_root,
    resolve_repo_root,
)


def _assert_non_path_lines_within_limit(text: str, *, repo_root: Path) -> None:
    for line in text.splitlines():
        if not line or line.startswith(str(repo_root)):
            continue
        assert len(line) <= 100


def test_set_workspace_imports_prerequisites(tmp_path: Path, capsys) -> None:
    clear_default_repo_root()
    set_workspace(tmp_path, refresh=False)

    repo_root = get_default_repo_root()
    data_raw = repo_root / "data_raw"
    summary_log = data_raw / "summary.log"
    exiobase_matching_dir = repo_root / "data_raw" / "mrio" / "exiobase_3" / "reg_matching"
    exiobase_aggregation_dir = data_raw / "mrio" / "exiobase_3" / "aggregation"
    carbon_cov_dir = data_raw / "mrio" / "exiobase_3" / "lcia" / "carbon_accounts_covs"
    oecd_aggregation_dir = data_raw / "mrio" / "oecd_v2025" / "aggregation"
    cc_notes_dir = data_raw / "carrying_capacities"
    methodological_notes_dir = data_raw / "methodological_notes"
    assert repo_root == resolve_repo_root(tmp_path)
    assert (cc_notes_dir / "pb_lcia_cc_steady_state.csv").exists()
    assert (methodological_notes_dir / "1_functional_units_and_allocation_methods.md").exists()
    assert (methodological_notes_dir / "fig-asocc-paths.svg").exists()
    assert (methodological_notes_dir / "methodological_note__acc_prospective.pdf").exists()
    assert (
        methodological_notes_dir / "methodological_note__asocc_fus_allocation_methods.pdf"
    ).exists()
    assert (methodological_notes_dir / "methodological_note__acc_uncertainty_sources.pdf").exists()
    assert (methodological_notes_dir / "methodological_note__steady_state__dynamic_cc.pdf").exists()
    assert (methodological_notes_dir / "recommended_citations.txt").exists()
    assert (exiobase_aggregation_dir / "agg_reg_template.csv").exists()
    assert (exiobase_aggregation_dir / "agg_reg_eu27.csv").exists()
    assert (exiobase_aggregation_dir / "agg_reg_world.csv").exists()
    assert (exiobase_aggregation_dir / "ixi" / "agg_sec_elec.csv").exists()
    assert (exiobase_aggregation_dir / "ixi" / "agg_sec_oecd_d.csv").exists()
    assert (oecd_aggregation_dir / "agg_reg_fr.csv").exists()
    assert (oecd_aggregation_dir / "agg_reg_world.csv").exists()
    assert (carbon_cov_dir / "reg_cbca_covs.csv").exists()
    assert (carbon_cov_dir / "reg_cbca_covs_agg_eu27.csv").exists()
    assert (carbon_cov_dir / "reg_cbca_covs_agg_world.csv").exists()
    assert (carbon_cov_dir / "README_agg_reg_and_group_indices_lcia_covs.txt").exists()
    assert (exiobase_matching_dir / "ssp_exiobase_3_matching.csv").exists()
    assert (exiobase_matching_dir / "wb_exiobase_3_matching.csv").exists()

    output = capsys.readouterr().out
    assert str(repo_root) in output
    assert str(summary_log) in output
    assert str(cc_notes_dir) in output
    assert str(methodological_notes_dir) in output
    assert output.count(str(cc_notes_dir)) == 1
    assert output.count(str(methodological_notes_dir)) == 1
    _assert_non_path_lines_within_limit(output, repo_root=repo_root)
    summary_text = summary_log.read_text(encoding="utf-8")
    assert summary_text.strip()
    assert str(repo_root) in summary_text
    assert str(cc_notes_dir) in summary_text
    assert str(methodological_notes_dir) in summary_text
    assert summary_text.count(str(cc_notes_dir)) == 1
    assert summary_text.count(str(methodological_notes_dir)) == 1
    _assert_non_path_lines_within_limit(summary_text, repo_root=repo_root)
    assert (cc_notes_dir / "README_add_custom_carrying_capacities.txt").exists()
    assert (
        repo_root
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "lcia"
        / "characterization_factors_matrices"
        / "README_add_custom_lcia_characterization_matrices.txt"
    ).exists()
    assert (
        repo_root
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "lcia"
        / "responsibility_periods"
        / "README_add_custom_lcia_responsibility_periods.txt"
    ).exists()
    assert packaged_prerequisites_root().name == "prerequisites"
    assert (
        packaged_prerequisites_root()
        .joinpath("mrio", "oecd_v2025", "aggregation", "agg_reg_fr.csv")
        .is_file()
    )
    assert (
        packaged_prerequisites_root()
        .joinpath("mrio", "exiobase_3", "aggregation", "ixi", "agg_sec_elec.csv")
        .is_file()
    )
    assert (
        packaged_prerequisites_root()
        .joinpath("mrio", "exiobase_3", "aggregation", "ixi", "agg_sec_oecd_d.csv")
        .is_file()
    )

    set_workspace(tmp_path, refresh=False)
    reuse_output = capsys.readouterr().out
    assert str(summary_log) in reuse_output
    _assert_non_path_lines_within_limit(reuse_output, repo_root=repo_root)

    summary_log.unlink()
    set_workspace(tmp_path, refresh=False)
    rewritten_output = capsys.readouterr().out
    assert str(summary_log) in rewritten_output
    assert str(cc_notes_dir) in rewritten_output
    _assert_non_path_lines_within_limit(rewritten_output, repo_root=repo_root)

    set_workspace(tmp_path, refresh=True)
    refresh_output = capsys.readouterr().out
    assert str(repo_root) in refresh_output
    assert str(summary_log) in refresh_output
    _assert_non_path_lines_within_limit(refresh_output, repo_root=repo_root)


def test_import_prereq_respects_refresh_flag(project_repo: Path) -> None:
    target = project_repo / "data_raw" / "carrying_capacities" / "pb_lcia_cc_steady_state.csv"
    bundled = (
        Path(__file__).resolve().parents[3]
        / "pyaesa"
        / "workspace_initialisation"
        / "prerequisites"
        / "carrying_capacities"
        / "pb_lcia_cc_steady_state.csv"
    ).read_text(encoding="utf-8")

    target.write_text("custom\n", encoding="utf-8")
    assert import_prerequisites(repo_root=project_repo, refresh=False) is False
    assert target.read_text(encoding="utf-8") == "custom\n"

    assert import_prerequisites(repo_root=project_repo, refresh=True) is True
    assert target.read_text(encoding="utf-8") == bundled


def test_project_output_root_requires_configured_workspace(tmp_path: Path) -> None:
    clear_default_repo_root()
    with pytest.raises(RuntimeError):
        get_default_repo_root()
    with pytest.raises(RuntimeError):
        project_outputs_root(project_name="demo")

    set_workspace(tmp_path, refresh=False)
    repo_root = get_default_repo_root()
    assert project_outputs_root(project_name="demo") == repo_root / "demo"
    with pytest.raises(ValueError):
        project_outputs_root(project_name="   ")


def test_set_workspace_rejects_blank_top_path() -> None:
    clear_default_repo_root()
    with pytest.raises(ValueError):
        set_workspace("", refresh=False)
    with pytest.raises(RuntimeError):
        get_default_repo_root()


def test_set_workspace_does_not_publish_repo_root_when_import_fails(tmp_path: Path) -> None:
    clear_default_repo_root()
    blocked_file_path = (
        resolve_repo_root(tmp_path)
        / "data_raw"
        / "carrying_capacities"
        / "pb_lcia_cc_steady_state.csv"
    )
    blocked_file_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        IsADirectoryError,
    ):
        set_workspace(tmp_path, refresh=True)

    with pytest.raises(RuntimeError):
        get_default_repo_root()
