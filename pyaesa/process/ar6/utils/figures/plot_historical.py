"""Historical only AR6 figure ownership."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyaesa.download.ar6.utils.config import (
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
)
from pyaesa.download.ar6.utils.sources import (
    ar6_historical_figure_reference,
    historical_sources,
)

from .plot_helpers import MT_TO_GT, numeric_year_columns


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric pandas Series for historical overlay inputs."""
    return pd.Series(pd.to_numeric(values, errors="raise"), copy=False)


def _read_historical_overlay(raw_data_dir: Path) -> pd.DataFrame:
    """Load the figure only AR6 historical comparison dataset."""
    overlay_file = raw_data_dir / historical_sources.AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME
    if not overlay_file.exists():
        raise RuntimeError(
            "The AR6 historical figure reference file is missing from the raw-data folder. "
            "Run download_ar6() to download the figure-only historical comparison dataset "
            "required by process_ar6(..., figures=True)."
        )
    try:
        return ar6_historical_figure_reference.read_ar6_historical_figure_reference(overlay_file)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc} Run download_ar6(refresh=True) to rebuild the AR6 historical "
            "comparison file used by process_ar6(..., figures=True)."
        ) from exc


def _overlay_series(overlay_df: pd.DataFrame, variable: str) -> pd.Series:
    """Return one numeric overlay series indexed by calendar year."""
    row_df = overlay_df.loc[
        (overlay_df["Model"] == "EDGAR")
        & (overlay_df["Scenario"] == "historical")
        & (overlay_df["Region"] == "World")
        & (overlay_df["Variable"] == variable),
        :,
    ]
    row = pd.Series(
        row_df.iloc[0].drop(labels=["Model", "Scenario", "Region", "Variable", "Unit"]),
        copy=False,
    )
    year_index = pd.Index(
        pd.Series(
            pd.to_numeric(pd.Series(row.index, copy=False), errors="raise"), copy=False
        ).astype(int)
    )
    series = _numeric_series(row)
    series.index = year_index
    return series.dropna().sort_index()


def _plot_overlay(
    ax: Axes,
    overlay_df: pd.DataFrame,
    *,
    variable: str,
    lower_variable: str,
    upper_variable: str,
    linestyle: str,
) -> None:
    """Plot the EDGAR historical comparison line and uncertainty band."""
    central = _overlay_series(overlay_df, variable)
    lower = _overlay_series(overlay_df, lower_variable)
    upper = _overlay_series(overlay_df, upper_variable)
    common_years = sorted(set(central.index).intersection(lower.index).intersection(upper.index))
    if not common_years:
        raise RuntimeError(
            "The AR6 historical figure reference file did not contain common "
            f"years for '{variable}'."
        )
    # The historical curves above are rendered with ``DataFrame.plot(...)``, which
    # uses positional x coordinates while formatting the ticks with the year
    # labels.
    # 1750 (starting year of EDGAR) is substracted to follow the same
    # x coordinate convention.
    x_values = [int(year) - 1750 for year in common_years]
    ax.plot(
        x_values,
        central.loc[common_years].to_numpy(dtype=float),
        linestyle=linestyle,
        color="red",
        alpha=0.25,
    )
    ax.fill_between(
        x_values,
        lower.loc[common_years].to_numpy(dtype=float),
        upper.loc[common_years].to_numpy(dtype=float),
        linestyle="solid",
        color="red",
        alpha=0.12,
    )


def plot_historical_emissions(
    historical_data_df: pd.DataFrame,
    raw_data_dir: Path,
) -> tuple[Figure, list[Axes]]:
    """Return the processed historical emissions figure."""
    overlay_df = _read_historical_overlay(raw_data_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=False)
    ax_ghg, ax_co2 = axes
    historic_colors = ["black", "gray", "lightgray", "darkgreen", "orange", "blue"]
    hist_years = [year for year in numeric_year_columns(historical_data_df) if 1750 <= year < 2024]
    if hist_years:
        (
            historical_data_df.loc[
                [NET_KYOTO_WITH_AFOLU, NET_KYOTO_WO_AFOLU],
                hist_years,
            ].T
            * MT_TO_GT
        ).plot(ax=ax_ghg, linestyle="solid", alpha=0.8, color=historic_colors[0:2])
        (
            historical_data_df.loc[
                [
                    "Emissions|Kyoto Gases|M.0.EL",
                    "Emissions|Kyoto Gases|M.LULUCF",
                    "Emissions|Kyoto Gases|M.AG",
                    "Emissions|CO2|Bunkers",
                ],
                hist_years,
            ].T
            * MT_TO_GT
        ).plot(ax=ax_ghg, linestyle="dashed", alpha=0.8, color=historic_colors[2:])
        (
            historical_data_df.loc[
                [NET_CO2_WITH_AFOLU, NET_CO2_WO_AFOLU],
                hist_years,
            ].T
            * MT_TO_GT
        ).plot(ax=ax_co2, linestyle="solid", alpha=0.8, color=historic_colors[0:2])
        (
            historical_data_df.loc[
                [
                    "Emissions|CO2|M.0.EL",
                    "Emissions|CO2|M.LULUCF",
                    "Emissions|CO2|M.AG",
                    "Emissions|CO2|Bunkers",
                ],
                hist_years,
            ].T
            * MT_TO_GT
        ).plot(ax=ax_co2, linestyle="dashed", alpha=0.8, color=historic_colors[2:])
    _plot_overlay(
        ax_ghg,
        overlay_df,
        variable="Emissions|Kyoto Gases (AR6-GWP100)",
        lower_variable="Emissions|Kyoto Gases (AR6-GWP100)|Lower",
        upper_variable="Emissions|Kyoto Gases (AR6-GWP100)|Upper",
        linestyle="solid",
    )
    _plot_overlay(
        ax_co2,
        overlay_df,
        variable="Emissions|CO2",
        lower_variable="Emissions|CO2|Lower",
        upper_variable="Emissions|CO2|Upper",
        linestyle="dashed",
    )
    ax_ghg.set_title("Historical GHG emissions")
    ax_ghg.set_ylabel("GtCO$_2$eq yr$^{-1}$")
    ax_ghg.grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    ax_co2.set_title("Historical CO2 emissions")
    ax_co2.set_ylabel("GtCO$_2$ yr$^{-1}$")
    ax_co2.grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    return fig, list(axes)
