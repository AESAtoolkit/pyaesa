import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.data import run_lcia as mod
from pyaesa.asocc.data.paths import _get_mrio_year_dir


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(message)


def _context(*, logger: _Logger, needs_lcia: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        source="oecd_v2025",
        group_version=None,
        needs_lcia=needs_lcia,
        lcia_methods=["gwp100_lcia"],
        fu_code="L1.a",
        selected_l1=["AR(E^{CBA_FD})"],
        selected_l2_one_step=[],
        combined=[],
        logger=logger,
    )


def _l2_context(*, logger: _Logger) -> SimpleNamespace:
    return SimpleNamespace(
        source="oecd_v2025",
        group_version=None,
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        fu_code="L2.a.a",
        selected_l1=[],
        selected_l2_one_step=["AR(E^{CBA_FD})"],
        combined=[],
        logger=logger,
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        lcia_metadata_cache={},
        lcia_available_years_cache={},
        lcia_method_payload_cache={},
        cf_by_method={},
        lcia_units={},
        skipped_years={},
        notices_emitted=set(),
        runtime_progress=None,
        runtime_source_prefix=None,
    )


def test_available_lcia_years_for_method_covers_cache_saved_dirs_and_metadata_contracts(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="run_lcia_years")
    logger = _Logger()
    context = _context(logger=logger)
    state = _state()

    assert mod._available_lcia_years_for_method(
        context=context,
        state=state,
        matrix_version=None,
        lcia_method="gwp100_lcia",
    ) == [2005, 2006]

    year_2006_dir = _get_mrio_year_dir(
        source="oecd_v2025",
        year=2006,
        group_version=None,
    )
    shutil.rmtree(year_2006_dir)
    assert mod._available_lcia_years_for_method(
        context=context,
        state=_state(),
        matrix_version=None,
        lcia_method="gwp100_lcia",
    ) == [2005]

    repo_with_invalid_token = allocation_dummy_repo_factory(name="run_lcia_year_tokens")
    payload_with_invalid_token = repo_with_invalid_token._read_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
    )
    payload_with_invalid_token["years"]["not_a_year"] = payload_with_invalid_token["years"]["2005"]
    repo_with_invalid_token._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version=None,
        payload=payload_with_invalid_token,
    )
    assert mod._available_lcia_years_for_method(
        context=context,
        state=_state(),
        matrix_version=None,
        lcia_method="gwp100_lcia",
    ) == [2005, 2006]


def test_load_lcia_for_year_covers_guards_fallback_skip_notice_and_cache(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="run_lcia_load")
    saved_dir_2005 = _get_mrio_year_dir(
        source="oecd_v2025",
        year=2005,
        group_version=None,
    )
    logger = _Logger()
    assert (
        mod._load_lcia_for_year(
            context=_context(logger=logger, needs_lcia=False),
            state=_state(),
            year=2005,
            saved_dir=saved_dir_2005,
        )
        is None
    )

    with pytest.raises(ValueError):
        mod._load_lcia_for_year(
            context=_context(logger=logger),
            state=_state(),
            year=2030,
            saved_dir=saved_dir_2005,
        )

    state = _state()
    method_year_out: dict[str, int] = {}
    first = mod._load_lcia_for_year(
        context=_context(logger=logger),
        state=state,
        year=2005,
        saved_dir=saved_dir_2005,
        method_year_out=method_year_out,
    )
    assert first is not None
    assert list(first) == ["gwp100_lcia"]
    assert method_year_out == {"gwp100_lcia": 2005}
    assert state.lcia_units["gwp100_lcia"].to_dict() == {"climate_parent": "kg CO2-eq / year"}

    cached_cf_state = _state()
    cached_cf_state.cf_by_method["gwp100_lcia"] = pd.Series(
        ["climate_parent"],
        index=pd.Index(["climate_child"], name="impact"),
    )
    cached_cf = mod._load_lcia_for_year(
        context=_context(logger=logger),
        state=cached_cf_state,
        year=2005,
        saved_dir=saved_dir_2005,
        group_version_override=None,
    )
    assert cached_cf is not None
    assert "gwp100_lcia" in cached_cf

    payload_dir = saved_dir_2005 / "enacting_metrics" / "level_1" / "gwp100_lcia"
    shutil.rmtree(payload_dir)
    second = mod._load_lcia_for_year(
        context=_context(logger=logger),
        state=state,
        year=2005,
        saved_dir=saved_dir_2005,
    )
    assert second is not None
    assert second["gwp100_lcia"] is first["gwp100_lcia"]

    allocation_dummy_repo_factory(name="run_lcia_l2")
    l2_loaded = mod._load_lcia_for_year(
        context=_l2_context(logger=_Logger()),
        state=_state(),
        year=2005,
        saved_dir=_get_mrio_year_dir(
            source="oecd_v2025",
            year=2005,
            group_version=None,
        ),
    )
    assert l2_loaded is not None
    assert "e_cba_fd_rp_sp" in l2_loaded["gwp100_lcia"]

    fallback_repo = allocation_dummy_repo_factory(name="run_lcia_fallback")
    fallback_repo.write_lcia_support(
        source="oecd_v2025",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=[2005],
    )
    fallback_saved_dir_2006 = _get_mrio_year_dir(
        source="oecd_v2025",
        year=2006,
        group_version=None,
    )
    fallback_state = _state()
    fallback_year_out: dict[str, int] = {}
    fallback = mod._load_lcia_for_year(
        context=_context(logger=_Logger()),
        state=fallback_state,
        year=2006,
        saved_dir=fallback_saved_dir_2006,
        allow_method_year_fallback=True,
        method_year_out=fallback_year_out,
    )
    assert fallback is not None
    assert fallback_year_out == {"gwp100_lcia": 2005}

    unavailable_repo = allocation_dummy_repo_factory(name="run_lcia_unavailable")
    unavailable_repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["gwp100_lcia"],
        available_years_by_method={"gwp100_lcia": []},
    )
    unavailable_saved_dir_2006 = _get_mrio_year_dir(
        source="oecd_v2025",
        year=2006,
        group_version=None,
    )
    unavailable_logger = _Logger()
    unavailable_state = _state()
    assert (
        mod._load_lcia_for_year(
            context=_context(logger=unavailable_logger),
            state=unavailable_state,
            year=2006,
            saved_dir=unavailable_saved_dir_2006,
            allow_method_year_fallback=False,
        )
        is None
    )
    assert set(unavailable_state.skipped_years[2006]) == {"gwp100_lcia"}
    assert unavailable_state.skipped_years[2006]["gwp100_lcia"]
    assert unavailable_logger.messages == []

    unavailable_with_notice_state = _state()
    assert (
        mod._load_lcia_for_year(
            context=_context(logger=unavailable_logger),
            state=unavailable_with_notice_state,
            year=2006,
            saved_dir=unavailable_saved_dir_2006,
            allow_method_year_fallback=True,
        )
        is None
    )
    assert set(unavailable_with_notice_state.skipped_years[2006]) == {"gwp100_lcia"}
    assert unavailable_with_notice_state.skipped_years[2006]["gwp100_lcia"]
    assert len(unavailable_logger.messages) == 1
    assert "gwp100_lcia" in unavailable_logger.messages[0]

    assert (
        mod._load_lcia_for_year(
            context=_context(logger=unavailable_logger),
            state=unavailable_with_notice_state,
            year=2006,
            saved_dir=unavailable_saved_dir_2006,
            allow_method_year_fallback=True,
        )
        is None
    )
    assert len(unavailable_logger.messages) == 1
