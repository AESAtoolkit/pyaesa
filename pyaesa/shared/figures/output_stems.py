"""Shared deterministic figure output stem normalization."""

from pyaesa.shared.figures.paths import strip_lcia_method_suffix


def dynamic_output_base_stem(*, base_stem: str, lcia_method: str) -> str:
    """Return one dynamic figure stem without a duplicated LCIA suffix."""
    return strip_lcia_method_suffix(stem=base_stem, lcia_methods=[lcia_method])
