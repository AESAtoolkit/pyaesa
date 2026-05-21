from types import SimpleNamespace

from pyaesa.asocc.orchestration.yearly.shared import output_spec_cache as cache_mod
from pyaesa.asocc.runtime.output.contracts import OutputRoute, OutputSpec


def _spec() -> OutputSpec:
    return OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem="demo_output",
        route=OutputRoute(
            level="level_2",
            bucket="l2_vs_global",
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p", "s_p"),
    )


def test_output_spec_cache_cover_missing_and_present_state() -> None:
    key = ("scope", "demo")
    spec = _spec()

    assert cache_mod.output_spec_cache_for_state(None) is None
    assert cache_mod.output_spec_cache_for_state(SimpleNamespace()) is None
    assert cache_mod.output_spec_cache_for_state(SimpleNamespace(output_spec_cache=[])) is None
    assert cache_mod.get_cached_output_spec(state=None, key=key) is None
    assert cache_mod.get_cached_output_spec(state=SimpleNamespace(), key=key) is None

    state = SimpleNamespace(output_spec_cache={})
    assert cache_mod.output_spec_cache_for_state(state) == {}
    assert cache_mod.get_cached_output_spec(state=state, key=key) is None

    cache_mod.set_cached_output_spec(state=None, key=key, spec=spec)
    cache_mod.set_cached_output_spec(state=SimpleNamespace(), key=key, spec=spec)

    cache_mod.set_cached_output_spec(state=state, key=key, spec=spec)
    assert cache_mod.get_cached_output_spec(state=state, key=key) == spec
