from pyaesa.asocc.orchestration import common_formatting as mod


def test_format_year_ranges_handles_empty_single_and_ranges() -> None:
    assert isinstance(mod.format_year_ranges([]), str)
    assert mod.format_year_ranges([2022]) == str(2022)

    compact_ranges = mod.format_year_ranges([2020, 2019, 2021, 2023, 2025, 2024])
    assert all(str(year) in compact_ranges for year in (2019, 2021, 2023, 2025))
    assert "2022" not in compact_ranges

    gapped_ranges = mod.format_year_ranges([1995, 1997, 1996, 2000])
    assert all(str(year) in gapped_ranges for year in (1995, 1997, 2000))
    assert "1998" not in gapped_ranges


def test_format_year_scope_handles_empty_and_pluralization() -> None:
    assert not mod.format_year_scope([])
    assert str(2030) in mod.format_year_scope([2030])

    scoped_years = mod.format_year_scope([2030, 2031, 2033])
    assert all(str(year) in scoped_years for year in (2030, 2031, 2033))
    assert "2032" not in scoped_years
