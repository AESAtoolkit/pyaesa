from pyaesa.asocc.orchestration.setup.formatting import formatting as mod


def test_setup_formatting_covers_multi_method_lcia_hint() -> None:
    assert mod._format_lcia_arg(["one_method"]) == "'one_method'"  # noqa: SLF001
    assert mod._format_lcia_arg(["z_method", "a_method", "z_method"]) == "['a_method', 'z_method']"  # noqa: SLF001
    assert (
        mod._process_mrio_hint(  # noqa: SLF001
            source="oecd_v2025",
            years=[2005],
            group_version=None,
            group_reg=False,
            group_sec=False,
            lcia_methods=["one_method"],
        )
        == "process_mrio(source='oecd_v2025', years=[2005], lcia_method='one_method')"
    )
    assert mod._process_mrio_hint(  # noqa: SLF001
        source="oecd_v2025",
        years=[2005, 2006],
        group_version="demo_reg",
        group_reg=True,
        group_sec=False,
        lcia_methods=["z_method", "a_method", "z_method"],
    ) == (
        "process_mrio(source='oecd_v2025', years=list(range(2005, 2007)), "
        "lcia_method=['a_method', 'z_method'], group_version='demo_reg', "
        "group_reg=True, group_sec=False)"
    )
