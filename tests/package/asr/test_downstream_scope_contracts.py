from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from pyaesa.asocc.runtime.scope.branch_resolution import (
    AsoccDeterministicPathScope,
    asocc_l1_dir,
    asocc_l2_dir,
    build_asocc_deterministic_path_scope,
    resolve_allocate_project_base,
)
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.shared.acc_asr_common.deterministic.downstream import inputs as inputs_mod
from pyaesa.shared.acc_asr_common.deterministic.downstream import (
    selection as selection_mod,
    shares as shares_mod,
    scenarios as downstream_scenarios_mod,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream import (
    tabular_io as tabular_io_mod,
)
from pyaesa.asr.deterministic.figures import groups as figure_groups_mod
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.shared.acc_asr_common.scope import composite as composite_mod
from pyaesa.asr.shared.lca import request as lca_mod
from pyaesa.asocc.runtime.paths.external import get_asocc_external_method_level_dir
from pyaesa.external_inputs.asocc.schema.file_specs import external_asocc_runtime_file_stem
from pyaesa.shared.figures import deterministic_transition_groups as shared_groups_mod
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.selectors import scenarios as shared_scenarios_mod


def _path_scope(tmp_path: Path) -> AsoccDeterministicPathScope:
    return build_asocc_deterministic_path_scope(
        proj_base=tmp_path,
        source_label="oecd_v2025",
        agg_version=None,
    )


def _loaded_asocc_share(
    *,
    file_stem: str,
    frame_wide: pd.DataFrame,
    relative_dir: Path = Path("level_2"),
) -> inputs_mod.LoadedAsoccShare:
    return inputs_mod.load_asocc_share(
        inputs_mod.AsoccShare(
            file_stem=file_stem,
            relative_dir=relative_dir,
            impacts=tuple(),
            source_label="native",
            frame_wide=frame_wide,
        )
    )


def _composite_base_allocate_args() -> dict[str, object]:
    mrio_scope = composite_mod.normalize_mrio_scope(
        source="oecd_v2025",
        agg_reg=False,
        agg_sec=False,
        agg_version="",
        group_indices=False,
    )
    return composite_mod.build_composite_base_allocate_args(
        project_name="downstream_scope_contracts",
        years=[2005, 2006],
        lcia_method=composite_mod.normalize_shared_lcia_methods("gwp100_lcia"),
        fu_code="L2.a.a",
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
        source=mrio_scope["source"],
        agg_reg=mrio_scope["agg_reg"],
        agg_sec=mrio_scope["agg_sec"],
        agg_version=mrio_scope["agg_version"],
        group_indices=mrio_scope["group_indices"],
        base_asocc_args=composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "l1_reg_aggreg": "post",
                "include_lcia_based_allocation_methods": False,
            },
            fu_code="L2.a.a",
        ),
    )


def test_downstream_tabular_covers_wrappers_and_write_formats(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "country": ["FR", "US"],
            "2005": [1.0, 2.0],
            "2006": [3.0, 4.0],
        }
    )
    assert tabular_io_mod.normalize_downstream_output_format(" csv ") == "csv"
    assert tabular_io_mod.detect_year_columns(frame) == ["2005", "2006"]
    assert tabular_io_mod.requested_year_columns(
        frame,
        requested_years=[2006, 2007],
    ) == ["2006"]
    assert tabular_io_mod.detect_id_columns(frame, ["2005", "2006"]) == ["country"]

    for output_format in ("csv", "pickle", "parquet"):
        path = tmp_path / f"table.{output_format}"
        tabular_io_mod.write_output_table(
            df=frame,
            output_path=path,
            output_format=output_format,
        )
        if output_format == "csv":
            round_trip = pd.read_csv(path)
        elif output_format == "pickle":
            round_trip = pd.read_pickle(path)
        else:
            round_trip = pd.read_parquet(path)
        pdt.assert_frame_equal(round_trip, frame)


def test_downstream_input_covers_file_loading_and_path_routing(
    tmp_path: Path,
) -> None:
    assert inputs_mod.collect_share_files(tmp_path) == []
    assert inputs_mod.collect_share_files(tmp_path / "missing") == []
    assert (
        inputs_mod.external_asocc_shares(
            proj_base=tmp_path,
            base_allocate_args=_composite_base_allocate_args(),
            fu_code="L2.a.a",
            external_method=None,
            years=[2005],
            lcia_method=None,
            output_source_label="oecd_v2025",
        )
        == []
    )

    csv_path = tmp_path / "nested" / "share.csv"
    pickle_path = tmp_path / "nested" / "share.pickle"
    parquet_path = tmp_path / "nested" / "share.parquet"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"country": ["FR"], "2005": [1.5]})
    frame.to_csv(csv_path, index=False)
    frame.to_pickle(pickle_path)
    frame.to_parquet(parquet_path, index=False)

    share_files = inputs_mod.collect_share_files(tmp_path)
    assert share_files == sorted([csv_path, parquet_path, pickle_path])

    assert inputs_mod.read_share_file(csv_path).equals(frame)
    assert inputs_mod.read_share_file(pickle_path).equals(frame)
    assert inputs_mod.read_share_file(parquet_path).equals(frame)
    with pytest.raises(ValueError):
        inputs_mod.read_share_file(tmp_path / "share.txt")

    wide_frame = pd.DataFrame({"value": [1.0]})
    asocc_share = inputs_mod.AsoccShare(
        file_stem="demo",
        relative_dir=Path("level_1"),
        impacts=tuple(),
        source_label="native",
        frame_wide=wide_frame,
    )
    read_frame = asocc_share.read()
    assert read_frame.equals(wide_frame)
    read_frame.loc[0, "value"] = 99.0
    assert float(wide_frame.loc[0, "value"]) == 1.0
    assert asocc_share.display_name == "demo.<external>"
    assert (
        inputs_mod.AsoccShare(
            file_stem="demo",
            relative_dir=Path("level_1"),
            impacts=tuple(),
            source_label="native",
            path=csv_path,
        ).display_name
        == csv_path.name
    )
    wide_out = asocc_share.frame_wide
    assert wide_out is not None
    assert float(wide_out.loc[0, "value"]) == 1.0
    assert inputs_mod._relative_share_path(
        tmp_path / "results" / "level_1" / "nested" / "share.csv"
    ) == Path("nested")  # noqa: SLF001
    assert inputs_mod._relative_share_path(
        tmp_path / "results" / "level_2" / "l2_vs_global" / "bucket" / "share.csv"
    ) == Path("bucket")  # noqa: SLF001
    assert inputs_mod._relative_share_path(
        tmp_path / "results" / "level_2" / "bucket" / "share.csv"
    ) == Path("bucket")  # noqa: SLF001
    assert inputs_mod._relative_share_path(tmp_path / "other" / "share.csv") == Path(".")  # noqa: SLF001
    path_only_share = inputs_mod.AsoccShare(
        file_stem="demo",
        relative_dir=Path("level_1"),
        impacts=tuple(),
        source_label="native",
        path=csv_path,
    )
    assert path_only_share.read().equals(frame)
    assert inputs_mod.asocc_share_reference_path(path_only_share) == csv_path
    assert inputs_mod.asocc_share_reference_path(asocc_share) == Path("demo.csv")


def test_downstream_asocc_shares_validate_canonical_method_identity() -> None:
    context = shares_mod.build_downstream_asocc_share_context(
        proj_base=Path("."),
        source_label="oecd_v2025",
        base_allocate_args={**_composite_base_allocate_args(), "ssp_scenario": ["SSP1"]},
        fu_code="L2.a.a",
        external_method=None,
        years=[2005, 2006],
        lcia_method="gwp100_lcia",
        output_source_label="native",
        branch_ssp_scenario=["SSP2"],
    )
    assert context.asocc_shares == []
    assert context.share_transition_meta == {}
    assert context.allowed_l1_l2_methods == set()
    assert shares_mod._resolved_downstream_scenario_tokens(  # noqa: SLF001
        base_allocate_args={"ssp_scenario": ["SSP1"]},
        branch_ssp_scenario=["SSP2", "SSP1"],
    ) == ["SSP1", "SSP2"]

    with pytest.raises(ValueError):
        selection_mod.asocc_share_declared_lcia_method(
            asocc_share=inputs_mod.AsoccShare(
                file_stem="demo__gwp100_lcia",
                relative_dir=Path("level_1"),
                impacts=tuple(),
                source_label="native",
                frame_wide=pd.DataFrame({"lcia_method": ["pb_lcia"]}),
            ),
            share_frame=pd.DataFrame({"lcia_method": ["pb_lcia"]}),
        )


def test_downstream_native_and_external_inputs_cover_real_routing(
    allocation_dummy_repo,
) -> None:
    repo_root = allocation_dummy_repo.repo_root
    path_scope = _path_scope(repo_root)
    l1_root = asocc_l1_dir(scope=path_scope, lcia_sub=None)
    l2_root = asocc_l2_dir(scope=path_scope, bucket="l2_vs_global", lcia_sub=None)
    (l1_root / "subdir").mkdir(parents=True, exist_ok=True)
    (l2_root / "subdir").mkdir(parents=True, exist_ok=True)
    l1_frame = pd.DataFrame({"2005": [1.0], "scenario": [None]})
    l2_frame = pd.DataFrame({"2005": [2.0], "scenario": [None]})
    l1_frame.to_csv(l1_root / "subdir" / "native_l1.csv", index=False)
    l2_frame.to_csv(l2_root / "subdir" / "native_l2.csv", index=False)

    base_allocate_args = _composite_base_allocate_args()
    native_l1 = inputs_mod.native_asocc_shares(
        proj_base=repo_root,
        source_label="oecd_v2025",
        fu_code="L1.a",
        base_allocate_args=base_allocate_args,
    )
    native_l2 = inputs_mod.native_asocc_shares(
        proj_base=repo_root,
        source_label="oecd_v2025",
        fu_code="L2.a.a",
        base_allocate_args=base_allocate_args,
    )
    native_all = inputs_mod.native_asocc_shares(
        proj_base=repo_root,
        source_label="oecd_v2025",
        fu_code="x",
        base_allocate_args=base_allocate_args,
    )
    assert [item.display_name for item in native_l1] == ["native_l1.csv"]
    assert [item.display_name for item in native_l2] == ["native_l2.csv"]
    assert {item.display_name for item in native_all} == {"native_l1.csv", "native_l2.csv"}

    external_dir = get_asocc_external_method_level_dir(
        proj_base=repo_root,
        storage_mode="deterministic",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    external_stem = external_asocc_runtime_file_stem(
        fu_code="L2.a.a",
        file_method_token="UT(FD)",
        l1_method=None,
        lcia_method=None,
        scenario=None,
    )
    external_frame = pd.DataFrame(
        {
            "r_p": ["FR"],
            "s_p": ["D"],
            "2005": [3.0],
        }
    )
    external_frame.to_csv(external_dir / f"{external_stem}.csv", index=False)

    external_asocc_shares = inputs_mod.external_asocc_shares(
        proj_base=repo_root,
        base_allocate_args=base_allocate_args,
        fu_code="L2.a.a",
        external_method={"one_step_methods": ["UT(FD)"]},
        years=[2005],
        lcia_method=None,
        output_source_label="oecd_v2025",
    )
    assert len(external_asocc_shares) == 1
    assert external_asocc_shares[0].source_label == "external"
    assert external_asocc_shares[0].read().shape[0] == 1

    combined = inputs_mod.combined_asocc_shares(
        proj_base=repo_root,
        source_label="oecd_v2025",
        base_allocate_args=base_allocate_args,
        fu_code="L2.a.a",
        external_method={"one_step_methods": ["UT(FD)"]},
        years=[2005],
        lcia_method=None,
        output_source_label="oecd_v2025",
    )
    assert len(combined) == len(native_l2) + len(external_asocc_shares)


def test_downstream_scenarios_cover_transition_and_suffix_logic() -> None:
    frame_wide = pd.DataFrame(
        {
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"],
            "2005": [None],
            "2010": [2.0],
        }
    )
    asocc_share = _loaded_asocc_share(
        file_stem="demo__ssp1",
        frame_wide=frame_wide,
    )
    assert downstream_scenarios_mod.asocc_share_ssp_scenario_labels(
        asocc_share,
    ) == {"SSP1"}
    assert (
        downstream_scenarios_mod.asocc_share_ssp_scenario_labels(
            _loaded_asocc_share(
                file_stem="demo",
                relative_dir=Path("level_1"),
                frame_wide=pd.DataFrame({"2005": [1.0]}),
            ),
        )
        == set()
    )

    metadata = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="demo",
                frame_wide=pd.DataFrame({"2005": [2.0]}),
            ),
            asocc_share,
            _loaded_asocc_share(
                file_stem="demo__l2_reuse_year_2030",
                frame_wide=pd.DataFrame({"2030": [1.0]}),
            ),
        ],
        scenario_tokens=["SSP2"],
    )
    assert metadata["demo__ssp1"]["base_stem"] == "demo"
    assert metadata["demo__ssp1"][ASOCC_SSP_SCENARIO_COLUMN] == "SSP1"
    assert metadata["demo__ssp1"]["asocc_ssp_scenario_labels"] == ["SSP1"]
    assert metadata["demo__ssp1"]["ssp_start_year"] == 2010
    assert metadata["demo__l2_reuse_year_2030"]["base_stem"] == "demo__l2_reuse_year_2030"
    assert metadata["demo__l2_reuse_year_2030"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert metadata["demo__l2_reuse_year_2030"]["asocc_ssp_scenario_labels"] == []
    assert metadata["demo__l2_reuse_year_2030"]["ssp_start_year"] is None

    metadata_reuse = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="demo__l2_reuse_year_2030",
                frame_wide=pd.DataFrame({"2030": [1.0]}),
            )
        ],
        scenario_tokens=["SSP2"],
    )
    assert metadata_reuse["demo__l2_reuse_year_2030"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert metadata_reuse["demo__l2_reuse_year_2030"]["ssp_start_year"] is None
    metadata_with_scenario = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="demo__ssp2",
                frame_wide=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"], "2005": [1.0]}),
            )
        ],
        scenario_tokens=["SSP2"],
    )
    assert metadata_with_scenario["demo__ssp2"][ASOCC_SSP_SCENARIO_COLUMN] == "SSP2"
    assert metadata_with_scenario["demo__ssp2"]["ssp_start_year"] == 2005

    no_scenario_column = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="plain",
                frame_wide=pd.DataFrame({"2005": [1.0]}),
            )
        ],
        scenario_tokens=["SSP1"],
    )
    assert no_scenario_column["plain"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert no_scenario_column["plain"]["ssp_start_year"] is None

    all_null_mask = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="plain_all_null",
                frame_wide=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: [None], "2005": [1.0]}),
            )
        ],
        scenario_tokens=["SSP1"],
    )
    assert all_null_mask["plain_all_null"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert all_null_mask["plain_all_null"]["ssp_start_year"] is None

    all_null_scenario_column = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="plain_null",
                frame_wide=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"], "2005": [None]}),
            )
        ],
        scenario_tokens=["SSP1"],
    )
    assert all_null_scenario_column["plain_null"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert all_null_scenario_column["plain_null"]["ssp_start_year"] is None

    suffix_without_requested_years = downstream_scenarios_mod.share_transition_metadata(
        asocc_shares=[
            _loaded_asocc_share(
                file_stem="suffix__ssp1",
                frame_wide=pd.DataFrame({"2005": [1.0]}),
            )
        ],
        scenario_tokens=["SSP1"],
    )
    assert suffix_without_requested_years["suffix__ssp1"][ASOCC_SSP_SCENARIO_COLUMN] is None
    assert suffix_without_requested_years["suffix__ssp1"]["ssp_start_year"] is None
    assert (
        downstream_scenarios_mod.share_transition_payload_for_output_stem(
            output_stem="",
            share_transition_meta=suffix_without_requested_years,
        )
        == {}
    )
    assert (
        downstream_scenarios_mod.share_transition_payload_for_output_stem(
            output_stem="suffix__ssp1",
            share_transition_meta=suffix_without_requested_years,
        )
        == suffix_without_requested_years["suffix__ssp1"]
    )

    assert (
        shared_groups_mod.normalize_companion_base_stem("demo__l2_reuse_year_2020")
        == "demo__l2_reuse_year_2020"
    )
    assert shared_groups_mod.normalize_companion_base_stem("demo__ssp1") == "demo"
    assert (
        shared_groups_mod.normalize_companion_base_stem("demo__l2_reuse_year_2020__ssp1")
        == "demo__l2_reuse_year_2020"
    )


def test_downstream_scope_normalization_covers_all_contract_branches(project_repo: Path) -> None:
    del project_repo
    assert shared_scenarios_mod.normalize_ssp_tokens(None) == []
    assert shared_scenarios_mod.normalize_ssp_tokens(["", "ssp2", "SSP1"]) == [
        "SSP1",
        "SSP2",
    ]
    assert shared_scenarios_mod.normalize_ssp_tokens([3]) == ["SSP3"]

    assert lca_mod.normalize_lca_args({"io_lca": {"active": True}}) == {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"type": "external"})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args(
            {
                "external_lca": {"active": True, "version_name": "supplier_v1"},
                "io_lca": {"active": True},
            }
        )
    assert lca_mod.normalize_lca_args({"external_lca": {"version_name": "supplier_v1"}}) == {
        "external_lca": {"active": True, "version_name": "supplier_v1"},
        "io_lca": {"active": False},
    }
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"external_lca": {"active": True}})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"external_lca": []})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"external_lca": {"extra": True}})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"external_lca": {"active": "yes"}})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args(
            {
                "external_lca": {"active": False, "version_name": "supplier_v1"},
                "io_lca": {"active": True},
            }
        )
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"io_lca": {"active": "yes"}})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"io_lca": []})
    with pytest.raises(ValueError):
        lca_mod.normalize_lca_args({"io_lca": {"extra": True}})

    assert composite_mod.normalize_shared_lcia_methods("gwp100_lcia") == ["gwp100_lcia"]
    with pytest.raises(ValueError):
        composite_mod.normalize_shared_lcia_methods([])
    normalized_mrio = composite_mod.normalize_mrio_scope(
        source="oecd_v2025",
        agg_reg=False,
        agg_sec=False,
        agg_version="v1",
        group_indices=False,
    )
    assert normalized_mrio["agg_version"] == "v1"
    assert (
        composite_mod.normalize_mrio_scope(
            source="oecd_v2025",
            agg_reg=False,
            agg_sec=False,
            agg_version=None,  # type: ignore[arg-type]
            group_indices=False,
        )["agg_version"]
        is None
    )
    with pytest.raises(ValueError):
        composite_mod.normalize_mrio_scope(
            source="oecd_v2025",
            agg_reg=1,  # type: ignore[arg-type]
            agg_sec=False,
            agg_version="",
            group_indices=False,
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args([], fu_code="L2.a.a")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        composite_mod.normalize_mrio_scope(
            source="oecd_v2025",
            agg_reg=False,
            agg_sec=False,
            agg_version="",
            group_indices="no",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_mrio_scope(
            source=None,  # type: ignore[arg-type]
            agg_reg=False,
            agg_sec=False,
            agg_version="",
            group_indices=False,
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_mrio_scope(
            source="oecd_v2025",
            agg_reg=False,
            agg_sec=False,
            agg_version=1,  # type: ignore[arg-type]
            group_indices=False,
        )

    normalized_base = composite_mod.normalize_base_asocc_args(
        {
            "method_plan": "one_step",
            "one_step_methods": "UT(FD)",
            "l1_reg_aggreg": "post",
            "include_lcia_based_allocation_methods": False,
        },
        fu_code="L2.a.a",
    )
    assert normalized_base["method_plan"] == "one_step"
    assert normalized_base["l1_reg_aggreg"] == "post"
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": 1,
                "l1_reg_aggreg": "post",
            },
            fu_code="L2.a.a",
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": [1],
                "l1_reg_aggreg": "post",
            },
            fu_code="L2.a.a",
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": [" "],
                "l1_reg_aggreg": "post",
            },
            fu_code="L2.a.a",
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "",
                "one_step_methods": ["UT(FD)"],
                "l1_reg_aggreg": "post",
            },
            fu_code="L2.a.a",
        )
    with pytest.raises(ValueError):
        composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "l1_reg_aggreg": "post",
                "unknown": True,
            },
            fu_code="L2.a.a",
        )
    assert (
        composite_mod.effective_asocc_lcia_methods(
            shared_lcia_methods=["gwp100_lcia"],
            include_lcia_based_allocation_methods=False,
        )
        is None
    )
    assert composite_mod.effective_asocc_lcia_methods(
        shared_lcia_methods=["gwp100_lcia"],
        include_lcia_based_allocation_methods=True,
    ) == ["gwp100_lcia"]

    request = composite_mod.build_composite_base_allocate_args(
        project_name="demo",
        years=[2005, 2006],
        lcia_method=composite_mod.normalize_shared_lcia_methods("gwp100_lcia"),
        fu_code="L2.a.a",
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
        source="oecd_v2025",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        group_indices=False,
        base_asocc_args=composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "l1_reg_aggreg": "post",
                "include_lcia_based_allocation_methods": False,
            },
            fu_code="L2.a.a",
        ),
    )
    assert request["project_name"] == "demo"
    assert request["source"] == "oecd_v2025"
    assert request["include_lcia_based_allocation_methods"] is False
    assert request["lcia_method"] is None
    assert composite_mod.asocc_lcia_methods_from_allocate_args(base_allocate_args=request) is None
    assert (
        resolve_allocate_project_base(
            base_allocate_args=normalize_base_allocate_args(
                composite_mod.base_asocc_kwargs_from_allocate_args(base_allocate_args=request)
            )
        ).name
        == "demo"
    )
    shared_asocc_request = composite_mod.build_composite_base_allocate_args(
        project_name="demo",
        years=[2005],
        lcia_method=["pb_lcia"],
        asocc_lcia_methods=["gwp100_lcia", "pb_lcia"],
        fu_code="L2.a.a",
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
        source="oecd_v2025",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        group_indices=False,
        base_asocc_args=composite_mod.normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "l1_reg_aggreg": "post",
            },
            fu_code="L2.a.a",
        ),
    )
    assert shared_asocc_request["lcia_method"] == ["gwp100_lcia", "pb_lcia"]
    assert composite_mod.asocc_lcia_methods_from_allocate_args(
        base_allocate_args=shared_asocc_request
    ) == ["gwp100_lcia", "pb_lcia"]


def test_downstream_year_selector_and_groups_cover_edge_contracts(tmp_path: Path) -> None:
    assert normalize_requested_years(2005) == [2005]

    assert (
        downstream_scenarios_mod._asocc_share_ssp_start_year(  # noqa: SLF001
            ssp_scenario="SSP2",
            ssp_scenario_labels={"SSP2"},
            covered_years={2005, 2006},
            historical_years=set(),
        )
        == 2005
    )
    assert (
        downstream_scenarios_mod._asocc_share_ssp_start_year(  # noqa: SLF001
            ssp_scenario="SSP2",
            ssp_scenario_labels={"SSP2"},
            covered_years={2005},
            historical_years={2005},
        )
        is None
    )
    assert (
        downstream_scenarios_mod.share_transition_payload_for_output_stem(
            output_stem=" ",
            share_transition_meta={"alpha": {"id": 1}},
        )
        == {}
    )
    assert downstream_scenarios_mod.share_transition_payload_for_output_stem(
        output_stem="alpha__beta__extra",
        share_transition_meta={
            "alpha": {"id": 1},
            "alpha__beta": {"id": 2},
        },
    ) == {"id": 2}

    with pytest.raises(ValueError):
        figure_groups_mod.resolve_scoped_figure_paths(
            root=tmp_path,
            output_paths=[tmp_path.parent / "outside.csv"],
            field_name="output_paths",
            family_label="ASR",
        )
    with pytest.raises(ValueError):
        figure_groups_mod.resolve_scoped_figure_paths(
            root=tmp_path,
            output_paths=[],
            field_name="output_paths",
            family_label="ASR",
        )
