"""Top level runner for population and GDP raw downloads.

This module centralises shared orchestration used by the individual downloaders
and provides a single entrypoint to run World Bank, IMF (Taiwan) and SSP
raw generation.
"""

import warnings

from pyaesa.download.pop_gdp.download_imf_twn import (
    _generate_imf_twn_raw,
)
from pyaesa.download.pop_gdp.download_ssp import (
    _generate_ssp_raw,
)
from pyaesa.download.pop_gdp.download_wb import (
    _generate_wb_raw,
)


def _run_download_pop_gdp(
    *,
    past_years: bool = True,
    future_years: bool = True,
    refresh: bool = False,
) -> None:
    """Run the population/GDP raw download pipeline."""
    if past_years:
        _generate_wb_raw(refresh=refresh)
        _generate_imf_twn_raw(refresh=refresh)

    if future_years:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"Passing literal json to 'read_json' is deprecated.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r"unique with argument that is not not a Series.*",
            )
            _generate_ssp_raw(refresh=refresh)


def download_pop_gdp(
    *,
    past_years: bool = True,
    future_years: bool = True,
    refresh: bool = False,
) -> None:
    """Download population/GDP datasets used by processing.

    Omit arguments to use their default.

    Args:
        past_years: If ``True``, include World Bank and IMF Taiwan historical
            population/GDP raw files. Default ``True``.
        future_years: If ``True``, include SSP future population/GDP raw
            files. Default ``True``.
        refresh: If ``True``, clear and rebuild only the selected raw
            population and GDP tables. ``past_years=True`` refreshes the World
            Bank and IMF Taiwan raw CSV files. ``future_years=True`` refreshes
            the SSP raw CSV file. Processed population and GDP outputs and
            project outputs are not refreshed. Defaults to ``False``.

    Returns:
        None.

    Raises:
        OSError: If writing a selected raw CSV or metadata file fails.
        RuntimeError: If retrieving or converting a selected upstream data
            source into the expected raw table fails.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.
        With ``refresh=False``, the function does not check whether more
        recent World Bank, IMF Taiwan, or SSP releases are available upstream.
        Run ``download_pop_gdp(refresh=True)`` when you intentionally want to
        pick up upstream revisions or newly covered years.

    Example:
        Download historical and future population/GDP inputs::

            from pyaesa import download_pop_gdp

            download_pop_gdp()
    """
    _run_download_pop_gdp(
        past_years=past_years,
        future_years=future_years,
        refresh=refresh,
    )
