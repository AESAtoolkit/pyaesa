from pyaesa.asocc.orchestration.setup.formatting import formatting as mod


def test_setup_formatting_covers_multi_method_lcia_hint() -> None:
    assert mod._format_lcia_arg(["one_method"]) == "'one_method'"  # noqa: SLF001
    assert mod._format_lcia_arg(["z_method", "a_method", "z_method"]) == "['a_method', 'z_method']"  # noqa: SLF001
    assert (
        mod._process_mrio_hint(  # noqa: SLF001
            source="oecd_v2025",
            years=[2005],
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
            lcia_methods=["one_method"],
        )
        == "process_mrio(source='oecd_v2025', years=[2005], lcia_method='one_method')"
    )
    assert mod._process_mrio_hint(  # noqa: SLF001
        source="oecd_v2025",
        years=[2005, 2006],
        agg_version="demo_reg",
        agg_reg=True,
        agg_sec=False,
        lcia_methods=["z_method", "a_method", "z_method"],
    ) == (
        "process_mrio(source='oecd_v2025', years=list(range(2005, 2007)), "
        "lcia_method=['a_method', 'z_method'], agg_version='demo_reg', "
        "agg_reg=True, agg_sec=False)"
    )
