from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.projection.reuse import (
    outputs as reuse_mod,
)


def _context(tmp_path: Path, *, output_format: str = "csv", intermediate_outputs: bool = True):
    return SimpleNamespace(
        output_format=output_format,
        proj_base=tmp_path,
        source="oecd_v2025",
        group_version="v1",
        group_reg=None,
        aggreg_indices=False,
        output_summed=False,
        l1_reg_aggreg="post",
        fu_code="L2.a.a",
        historical_years=[2018, 2019, 2020],
        intermediate_outputs=intermediate_outputs,
    )


def _state():
    return SimpleNamespace(
        ut_reuse_preweight_cache={},
        ut_reuse_one_step_cache={},
    )


def _write_wide_output(*, context, bucket: str, file_stem: str, frame: pd.DataFrame) -> Path:
    path = reuse_mod.reuse_output_path_for(context=context, bucket=bucket, file_stem=file_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    if context.output_format == "csv":
        frame.to_csv(path, index=False)
    elif context.output_format == "pickle":
        frame.to_pickle(path)
    else:
        frame.to_parquet(path)
    return path


def test_reuse_output_and_source_loading(tmp_path: Path) -> None:
    assert reuse_mod._normalized_reuse_lcia_key(l2_method="UT(FD)", lcia_key="IPCC") is None
    assert reuse_mod._normalized_reuse_lcia_key(l2_method="AR(E)", lcia_key="IPCC") == "IPCC"
    assert reuse_mod._reuse_file_stem(bucket="l2_in_l1", l2_method="UT(FDa)", lcia_key=None) == (
        "l2_UT(FDa)"
    )
    assert reuse_mod._reuse_file_stem(bucket="l2_vs_global", l2_method="UT(FD)", lcia_key=None) == (
        "UT(FD)"
    )
    assert reuse_mod._is_historical_year(context=_context(tmp_path), year=2019) is True
    assert reuse_mod._is_historical_year(context=_context(tmp_path), year=2030) is False
    assert reuse_mod._year_columns(pd.DataFrame({"2020": [1], "x": [2], 2030: [3]})) == [
        "2020",
        "2030",
    ]

    one_year = pd.DataFrame({"2030": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    normalized = reuse_mod._normalize_single_year_frame(frame=one_year, source_year=2018)
    assert list(normalized.columns) == [2018]
    with pytest.raises(ValueError):
        reuse_mod._normalize_single_year_frame(
            frame=pd.DataFrame({"2030": [1.0], "2031": [2.0]}),
            source_year=2018,
        )

    context = _context(tmp_path, output_format="csv")
    out_path = reuse_mod.reuse_output_path_for(context=context, bucket="l2_in_l1", file_stem="x")
    assert out_path.name == "x.csv"

    csv_path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)
    assert int(reuse_mod._read_wide_output(path=csv_path, output_format="csv").iloc[0, 0]) == 1
    pickle_path = tmp_path / "x.pickle"
    pd.DataFrame({"a": [2]}).to_pickle(pickle_path)
    assert (
        int(reuse_mod._read_wide_output(path=pickle_path, output_format="pickle").iloc[0, 0]) == 2
    )
    parquet_path = tmp_path / "x.parquet"
    pd.DataFrame({"a": [3]}).to_parquet(parquet_path)
    assert (
        int(reuse_mod._read_wide_output(path=parquet_path, output_format="parquet").iloc[0, 0]) == 3
    )

    missing_context = _context(tmp_path)
    with pytest.raises(FileNotFoundError):
        reuse_mod._load_source_year_from_output(
            context=missing_context,
            bucket="l2_vs_global",
            file_stem="UT(FD)",
            source_year=2019,
        )

    _write_wide_output(
        context=missing_context,
        bucket="l2_vs_global",
        file_stem="UT(FD)",
        frame=pd.DataFrame({"l2_method": ["UT(FD)"], "r_p": ["FR"], "2018": [1.0]}),
    )
    with pytest.raises(ValueError):
        reuse_mod._load_source_year_from_output(
            context=missing_context,
            bucket="l2_vs_global",
            file_stem="UT(FD)",
            source_year=2019,
        )

    _write_wide_output(
        context=missing_context,
        bucket="l2_vs_global",
        file_stem="UT(FD_dup)",
        frame=pd.DataFrame(
            {
                "l2_method": ["UT(FD)", "UT(FD)"],
                "r_p": ["FR", "FR"],
                "2019": [1.0, 1.0],
            }
        ),
    )
    loaded = reuse_mod._load_source_year_from_output(
        context=missing_context,
        bucket="l2_vs_global",
        file_stem="UT(FD_dup)",
        source_year=2019,
    )
    assert list(loaded.columns) == [2019]
    assert list(loaded.index.names) == ["r_p"]

    _write_wide_output(
        context=missing_context,
        bucket="l2_vs_global",
        file_stem="UT(FD_unique)",
        frame=pd.DataFrame(
            {
                "l2_method": ["UT(FD)", "UT(FD)"],
                "r_p": ["FR", "US"],
                "2019": [1.0, 2.0],
            }
        ),
    )
    unique_loaded = reuse_mod._load_source_year_from_output(
        context=missing_context,
        bucket="l2_vs_global",
        file_stem="UT(FD_unique)",
        source_year=2019,
    )
    assert set(unique_loaded.index.tolist()) == {"FR", "US"}


def test_reuse_preweight_compute_cache_and_loaders(tmp_path: Path) -> None:
    state = _state()
    context = _context(tmp_path)
    frame = pd.DataFrame({"2020": [1.0]}, index=pd.Index(["FR"], name="r_p"))

    reuse_mod.cache_historical_preweight(
        context=context,
        state=state,
        year=2019,
        l2_method="UT(FDa)",
        lcia_key="IPCC",
        frame=frame,
    )
    assert ("preweight", "UT(FDa)", None, 2019) in state.ut_reuse_preweight_cache

    reuse_mod.cache_historical_one_step_result(
        context=context,
        state=state,
        year=2019,
        l2_method="UT(FD)",
        lcia_key="IPCC",
        frame=frame,
    )
    assert ("one_step", "UT(FD)", None, 2019) in state.ut_reuse_one_step_cache

    # Non historical years should not be cached.
    reuse_mod.cache_historical_preweight(
        context=context,
        state=state,
        year=2030,
        l2_method="UT(FDa)",
        lcia_key="IPCC",
        frame=frame,
    )
    reuse_mod.cache_historical_one_step_result(
        context=context,
        state=state,
        year=2030,
        l2_method="UT(FD)",
        lcia_key="IPCC",
        frame=frame,
    )
    assert ("preweight", "UT(FDa)", None, 2030) not in state.ut_reuse_preweight_cache
    assert ("one_step", "UT(FD)", None, 2030) not in state.ut_reuse_one_step_cache

    # load_reuse_preweight cache hit
    cached = pd.DataFrame({2019: [1.0]}, index=pd.Index(["FR"], name="r_p"))
    state.ut_reuse_preweight_cache[("preweight", "UT(FDa)", None, 2019)] = cached
    assert (
        reuse_mod.load_reuse_preweight(
            context=context,
            state=state,
            l2_method="UT(FDa)",
            lcia_key="IPCC",
            l2_reuse_year=2019,
        )
        is cached
    )

    # load from outputs
    state.ut_reuse_preweight_cache.clear()
    _write_wide_output(
        context=context,
        bucket="l2_in_l1",
        file_stem="l2_UT(FDa)",
        frame=pd.DataFrame({"l2_method": ["UT(FDa)"], "r_p": ["FR"], "2019": [3.0]}),
    )
    loaded = reuse_mod.load_reuse_preweight(
        context=context,
        state=state,
        l2_method="UT(FDa)",
        lcia_key="IPCC",
        l2_reuse_year=2019,
    )
    assert float(loaded.iloc[0, 0]) == 3.0

    # Missing l2_in_l1 source output is always an error (strict no fallback).
    state.ut_reuse_preweight_cache.clear()
    with pytest.raises(FileNotFoundError):
        reuse_mod.load_reuse_preweight(
            context=_context(tmp_path / "missing"),
            state=state,
            l2_method="UT(FDa)",
            lcia_key="IPCC",
            l2_reuse_year=2019,
        )

    # one step reuse loader remaps year columns and uses cache.
    state.ut_reuse_one_step_cache.clear()
    _write_wide_output(
        context=context,
        bucket="l2_vs_global",
        file_stem="UT(FD)",
        frame=pd.DataFrame({"l2_method": ["UT(FD)"], "r_p": ["FR"], "2019": [5.0]}),
    )
    one_step = reuse_mod.load_reuse_one_step_result(
        context=context,
        state=state,
        l2_method="UT(FD)",
        lcia_key="IPCC",
        l2_reuse_year=2019,
        target_year=2030,
    )
    assert list(one_step.columns) == [2030]
    one_step_cached = reuse_mod.load_reuse_one_step_result(
        context=context,
        state=state,
        l2_method="UT(FD)",
        lcia_key="IPCC",
        l2_reuse_year=2019,
        target_year=2031,
    )
    assert list(one_step_cached.columns) == [2031]
