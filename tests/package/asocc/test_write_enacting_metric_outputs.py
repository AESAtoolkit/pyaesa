from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.io.metadata import EnactingMetricKey
from pyaesa.asocc.orchestration.write.writers import enacting_metric as mod
from pyaesa.asocc.orchestration.write.writers.progress import tick_write_progress
from pyaesa.asocc.orchestration.write.writers.ut_gvaa_identity_closure import (
    write_ut_gvaa_identity_closure_audit,
)
from pyaesa.asocc.runtime.output.contracts import OutputArtifact


def _projection_context(*, mode: str = "regression") -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        mode=mode,
        reg_window=(2018, 2021),
        l2_reuse_years=(2030,),
        is_future_year=lambda year: int(year) > 2025,
        route_for_l2_method=lambda l2_method: (
            "regression" if l2_method == "fd_rf" else "historical_reuse"
        ),
    )


def _context(
    *,
    projection_context: SimpleNamespace | None = None,
    fu_code: str = "L1.a",
) -> SimpleNamespace:
    return SimpleNamespace(
        proj_base=Path("C:/tmp/write_enacting_metric"),
        source="oecd_v2025",
        fu_code=fu_code,
        group_version=None,
        group_reg=False,
        aggreg_indices=False,
        l1_reg_aggreg="post",
        output_format="csv",
        intermediate_outputs=True,
        projection_context=projection_context,
        wb_df=pd.DataFrame(columns=["2019"]),
        wb_df_raw=pd.DataFrame(columns=["variable", "unit"]),
        ssp_df_raw=pd.DataFrame(columns=["variable", "unit", "scenario"]),
        persisted_years=[2005, 2030],
        resolved_years=[2005, 2030],
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        mrio_default_monetary_unit="M EUR",
        mrio_units={"fd_rf": "M EUR"},
        lcia_units={
            "gwp100_lcia": pd.Series(
                ["kgCO2e"],
                index=pd.Index(["Climate"], name="impact"),
            )
        },
        enacting_metric_levels={},
        enacting_metric_inputs={},
    )


def test_write_progress_caps_current_and_uses_default_label(capsys) -> None:
    state = SimpleNamespace(write_progress_total=1, write_progress_current=1)
    tick_write_progress(context=SimpleNamespace(source="oecd_v2025"), state=state)

    out = capsys.readouterr().out
    assert out
    assert state.write_progress_current == 1
    assert state.write_progress_last_width == 0


def test_ut_gvaa_identity_closure_audit_covers_empty_and_append_paths(tmp_path: Path) -> None:
    context = _context()
    context.proj_base = tmp_path
    context.output_source = context.source
    state = SimpleNamespace(
        ut_gvaa_identity_closure_rows=[],
        write_progress_total=1,
        write_progress_current=0,
        runtime_progress=None,
    )
    assert (
        write_ut_gvaa_identity_closure_audit(
            context=context,
            state=state,
            refresh_effective=False,
        )
        is None
    )

    state.ut_gvaa_identity_closure_rows = [{}]
    assert (
        write_ut_gvaa_identity_closure_audit(
            context=context,
            state=state,
            refresh_effective=False,
        )
        is None
    )

    state.ut_gvaa_identity_closure_rows = [
        {
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "year": 2005,
            "ssp_scenario": None,
            "l2_method": "UT(GVAa)",
            "comparator_method": "UT(GVA)",
            "l1_method": None,
            "impact": None,
            "lcia_key": None,
            "reference_year": None,
            "l2_reuse_year": None,
            "r_p": "FR",
            "s_p": "A",
            "ut_gvaa_raw": 0.4,
            "ut_gva_floor": 0.5,
            "ut_gvaa_final": 0.5,
            "delta_added": 0.1,
            "adjustment_note": "floor",
        }
    ]
    path = write_ut_gvaa_identity_closure_audit(
        context=context,
        state=state,
        refresh_effective=True,
    )
    assert path is not None and path.exists()
    state.ut_gvaa_identity_closure_rows[0]["ut_gvaa_final"] = 0.6
    write_ut_gvaa_identity_closure_audit(
        context=context,
        state=state,
        refresh_effective=False,
    )
    rows = pd.read_csv(path)
    assert rows["ut_gvaa_final"].tolist() == [0.6]


def test_reset_metric_index_and_build_output_cover_level_routing(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"2005": [1.0], "2030": [2.0]},
        index=pd.Index(["FR"], name="r_f"),
    )
    reset = mod._reset_metric_index_strict(frame)  # noqa: SLF001
    assert list(reset.columns) == ["r_f", "2005", "2030"]
    multi_reset = mod._reset_metric_index_strict(  # noqa: SLF001
        pd.DataFrame(
            {"2005": [1.0], "2030": [2.0]},
            index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_f", "s_p"]),
        )
    )
    assert list(multi_reset.columns) == ["r_f", "s_p", "2005", "2030"]

    with pytest.raises(ValueError):
        mod._reset_metric_index_strict(pd.DataFrame({"2005": [1.0]}))  # noqa: SLF001
    with pytest.raises(ValueError):
        mod._reset_metric_index_strict(  # noqa: SLF001
            pd.DataFrame({"unit": [1.0]}, index=pd.Index(["FR"], name="unit"))
        )

    context = _context()
    context.proj_base = tmp_path
    state = _state()
    key = EnactingMetricKey(metric="fd_rf")
    state.enacting_metric_levels[key] = "level_1"
    artifact, out_path = mod._build_enacting_metric_output(  # noqa: SLF001
        context=context,
        state=state,
        key=key,
        year_map={
            2005: pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            2030: pd.Series([2.0], index=pd.Index(["FR"], name="r_f")),
        },
        l1_source="native_source",
        projection_subfolder=None,
    )
    assert isinstance(artifact, OutputArtifact)
    assert artifact.schema.columns == ("unit", "r_f")
    assert out_path.name == "fd_rf.csv"
    assert out_path.parts[-3:] == ("results", "enacting_metrics", "fd_rf.csv")

    artifact_default_source, out_path_default_source = mod._build_enacting_metric_output(  # noqa: SLF001
        context=context,
        state=state,
        key=key,
        year_map={
            2005: pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
        },
        l1_source=None,
        projection_subfolder=None,
    )
    assert isinstance(artifact_default_source, OutputArtifact)
    assert out_path_default_source.name == "fd_rf.csv"
    assert context.source in str(out_path_default_source)

    context.fu_code = "L2.a.a"
    key_lcia = EnactingMetricKey(metric="e_cba_fd_reg", lcia_method="gwp100_lcia")
    state.enacting_metric_levels[key_lcia] = "level_2"
    artifact_lcia, out_path_lcia = mod._build_enacting_metric_output(  # noqa: SLF001
        context=context,
        state=state,
        key=key_lcia,
        year_map={
            2005: pd.Series([1.0], index=pd.Index(["Climate"], name="impact")),
            2030: pd.Series([2.0], index=pd.Index(["Climate"], name="impact")),
        },
        l1_source=None,
        projection_subfolder="regression_proj",
    )
    assert isinstance(artifact_lcia, OutputArtifact)
    assert artifact_lcia.schema.columns == ("unit", "impact")
    assert out_path_lcia.name == "e_cba_fd_reg_gwp100_lcia.csv"
    assert out_path_lcia.parts[-5:] == (
        "results",
        "level_2",
        "enacting_metrics",
        "regression_proj",
        "e_cba_fd_reg_gwp100_lcia.csv",
    )


def test_projection_split_and_output_target_count_cover_routing_branches() -> None:
    year_map = {
        2005: pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
        2030: pd.Series([2.0], index=pd.Index(["FR"], name="r_f")),
    }
    key = EnactingMetricKey(metric="fd_rf")

    assert mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=None),
        key=key,
        year_map=year_map,
    ) == [(None, year_map)]
    assert mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=_projection_context(mode="historical")),
        key=key,
        year_map=year_map,
    ) == [(None, year_map)]
    assert mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=_projection_context()),
        key=EnactingMetricKey(metric="population"),
        year_map=year_map,
    ) == [(None, year_map)]

    split = mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=_projection_context()),
        key=key,
        year_map=year_map,
    )
    assert [item[0] for item in split] == [None, "regression_proj"]
    assert split[0][1] == {2005: year_map[2005]}
    assert split[1][1] == {2030: year_map[2030]}
    assert mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=_projection_context()),
        key=key,
        year_map={2005: year_map[2005]},
    ) == [(None, {2005: year_map[2005]})]
    assert mod._split_year_map_for_output(  # noqa: SLF001
        context=_context(projection_context=_projection_context()),
        key=key,
        year_map={2030: year_map[2030]},
    ) == [("regression_proj", {2030: year_map[2030]})]

    state = SimpleNamespace(
        enacting_metric_inputs={
            EnactingMetricKey(metric="fd_rf"): year_map,
            EnactingMetricKey(metric="population"): {},
        }
    )
    assert (
        mod.count_enacting_metric_output_targets(  # noqa: SLF001
            context=_context(projection_context=_projection_context()),
            state=state,
        )
        == 2
    )
    assert (
        mod.count_enacting_metric_output_targets(  # noqa: SLF001
            context=SimpleNamespace(intermediate_outputs=False),
            state=state,
        )
        == 0
    )


def test_write_enacting_metric_outputs_covers_noop_and_writer_calls(tmp_path: Path) -> None:
    key = EnactingMetricKey(metric="fd_rf")
    state = _state()
    state.enacting_metric_levels[key] = "level_2"
    state.enacting_metric_inputs = {
        EnactingMetricKey(metric="population"): {},
        key: {
            2005: pd.Series([1.0], index=pd.Index(["FR"], name="r_f")),
            2030: pd.Series([2.0], index=pd.Index(["FR"], name="r_f")),
        },
    }
    context = _context(projection_context=_projection_context(), fu_code="L2.a.a")
    context.proj_base = tmp_path

    calls: list[tuple[Path, str | None]] = []

    def _record_writer(**kwargs) -> None:
        calls.append((kwargs["out_path"], kwargs["artifact"].schema.columns[-1]))

    mod._write_enacting_metric_outputs(  # noqa: SLF001
        context=SimpleNamespace(intermediate_outputs=False),
        state=state,
        refresh_effective=False,
        l1_source=None,
        write_result_artifact=_record_writer,
    )
    assert calls == []

    mod._write_enacting_metric_outputs(  # noqa: SLF001
        context=context,
        state=state,
        refresh_effective=True,
        l1_source="native_source",
        write_result_artifact=_record_writer,
    )
    assert len(calls) == 2
    assert any(path.parent.name == "regression_proj" for path, _ in calls)
    assert any(path.parent.name == "enacting_metrics" for path, _ in calls)
