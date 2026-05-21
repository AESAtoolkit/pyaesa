import re

from pyaesa.asocc.orchestration.setup.request.selection import (
    build_indices_tag,
)

INVALID_WINDOWS_CHARS = set('<>:"/\\|?*')


def _segments(tag: str) -> list[str]:
    return [piece for piece in tag.split("__") if piece]


def test_build_indices_tag_all_indices_when_no_filters() -> None:
    tag = build_indices_tag({"r_p": None, "s_p": None, "r_c": None, "r_f": None})
    assert tag == "all_indices"


def test_build_indices_tag_windows_safe_for_invalid_chars() -> None:
    tag = build_indices_tag(
        {
            "r_p": None,
            "s_p": [
                "Incineration of waste: Plastic",
                "Landfill of waste: Plastic",
                "Manufacture of rubber and plastic products (25)",
            ],
            "r_c": None,
            "r_f": None,
        }
    )
    assert tag != "all_indices"
    assert not any(char in tag for char in INVALID_WINDOWS_CHARS)
    for seg in _segments(tag):
        assert not seg.endswith((" ", "."))
        assert re.fullmatch(r"[A-Za-z0-9._()+-]+", seg)


def test_build_indices_tag_is_stable_for_order_and_duplicates() -> None:
    a = build_indices_tag(
        {
            "r_p": ["USA", "FRA", "USA"],
            "s_p": ["A", "B"],
            "r_c": None,
            "r_f": None,
        }
    )
    b = build_indices_tag(
        {
            "r_p": ["FRA", "USA"],
            "s_p": ["B", "A", "A"],
            "r_c": None,
            "r_f": None,
        }
    )
    assert a == b


def test_build_indices_tag_segment_length_is_bounded() -> None:
    long_values = [f"sector_name_with_very_long_text_{i}_{'x' * 50}" for i in range(30)]
    tag = build_indices_tag({"r_p": None, "s_p": long_values, "r_c": None, "r_f": None})
    for seg in _segments(tag):
        assert len(seg) <= 120


def test_build_indices_tag_keeps_readable_tokens_for_five_s_p_values() -> None:
    tag = build_indices_tag(
        {
            "r_p": None,
            "s_p": [
                "Plastics, basic",
                "Re-processing of secondary plastic into new plastic",
                "Manufacture of rubber and plastic products (25)",
                "Incineration of waste: Plastic",
                "Landfill of waste: Plastic",
            ],
            "r_c": None,
            "r_f": None,
        }
    )
    assert tag.startswith("s_p-")
    assert not tag.startswith("s_p-n5_")
    assert "Plastic" in tag or "Plastics" in tag
