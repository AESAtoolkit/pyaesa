"""Historical emissions processing for AR6 harmonization."""

import numpy as np
import pandas as pd

from pyaesa.download.ar6.utils.sources import (
    historical_sources,
)
from pyaesa.download.ar6.utils.config import (
    GROSS_ALT_CO2_WITH_AFOLU,
    GROSS_ALT_CO2_WO_AFOLU,
    GROSS_ALT_KYOTO_WITH_AFOLU,
    GROSS_ALT_KYOTO_WO_AFOLU,
    GROSS_CO2_WITH_AFOLU,
    GROSS_CO2_WO_AFOLU,
    GROSS_KYOTO_WITH_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
)

# This workflow converts the GCP territorial bunkers carbon series to CO2 mass
# with a factor of 3.664 before it is added to the PRIMAP based historical baseline.
CARBON_TO_CO2_MASS_RATIO = 3.664
PRIMAP_KT_TO_MT = 1e-3


def _require_row_series(data_df: pd.DataFrame, row_label: str, *, context: str) -> pd.Series:
    """Return the row series for one historical baseline variable."""
    del context
    return data_df.loc[row_label]


def _add_year_values(
    data_df: pd.DataFrame,
    *,
    row_label: str,
    years: list[int],
    values: np.ndarray,
) -> None:
    """Add numeric values to one existing yearly row."""
    row = _require_row_series(data_df, row_label, context="Historical baseline update").copy()
    year_index = pd.Index(years)
    current_values = pd.Series(
        pd.to_numeric(row.loc[year_index], errors="raise"),
        index=year_index,
        dtype=float,
    )
    if current_values.isna().any():
        raise RuntimeError(
            f"Historical baseline row '{row_label}' contains missing values in shared years."
        )
    row.loc[year_index] = current_values.to_numpy(dtype=float) + values
    data_df.loc[row_label] = row


def _set_year_values(
    data_df: pd.DataFrame,
    *,
    row_label: str,
    years: list[int],
    values: np.ndarray,
) -> None:
    """Write numeric values to one yearly row, creating it when needed."""
    year_labels = list(years)
    if row_label in data_df.index:
        row = _require_row_series(
            data_df,
            row_label,
            context="Historical baseline write",
        ).copy()
    else:
        row = pd.Series(np.nan, index=data_df.columns, dtype=float)
    row.loc[year_labels] = values.tolist()
    data_df.loc[row_label] = row


def process_historical_emissions(folder_path_s: str):
    """Build the historical harmonization baseline from PRIMAP and GCP raw files."""
    data_path_gcp = f"{folder_path_s}{historical_sources.GCP_NATIONAL_FOSSIL_LOCAL_NAME}"
    gcp_data_df = pd.read_excel(
        data_path_gcp,
        sheet_name="Territorial Emissions",
        engine="calamine",
    )
    year_col = gcp_data_df.iloc[:, 0]
    gcp_header_row = gcp_data_df.iloc[9]
    bunkers_col = gcp_header_row.loc[gcp_header_row == "Bunkers"].index[0]
    first_valid_year_idx = year_col.loc[year_col == 1850].index[0]
    gcp_years_series = pd.Series(year_col.loc[first_valid_year_idx:], dtype=int)
    years_valid_index = gcp_years_series.index
    years_index_gcp = gcp_years_series.to_numpy(dtype=int)
    bunkers_co2_gcp_df = pd.DataFrame(
        # The GCP territorial bunkers sheet reports carbon mass, while the
        # historical baseline is stored in CO2 mass units.
        gcp_data_df.loc[years_valid_index, bunkers_col].to_numpy(dtype=float)
        * CARBON_TO_CO2_MASS_RATIO,
        index=years_index_gcp,
        columns=["Emissions|CO2|Bunkers"],
    ).T
    bunkers_co2_gcp_df.insert(0, column="units", value="MtCO2/yr")
    data_path_excl = f"{folder_path_s}{historical_sources.PRIMAP_FINAL_LOCAL_NAME}"
    data_path_incl = f"{folder_path_s}{historical_sources.PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME}"
    primap_excl_df = pd.read_csv(data_path_excl, index_col=[0, 1, 2, 3, 4, 5, 6])
    primap_incl_df = pd.read_csv(data_path_incl, index_col=[0, 1, 2, 3, 4, 5, 6])
    my_index = [
        NET_KYOTO_WITH_AFOLU,
        NET_KYOTO_WO_AFOLU,
        NET_CO2_WITH_AFOLU,
        NET_CO2_WO_AFOLU,
        GROSS_KYOTO_WITH_AFOLU,
        GROSS_KYOTO_WO_AFOLU,
        GROSS_CO2_WITH_AFOLU,
        GROSS_CO2_WO_AFOLU,
        GROSS_ALT_KYOTO_WITH_AFOLU,
        GROSS_ALT_KYOTO_WO_AFOLU,
        GROSS_ALT_CO2_WITH_AFOLU,
        GROSS_ALT_CO2_WO_AFOLU,
    ]
    primap_years = primap_excl_df.columns.astype(int).tolist()
    historic_emissions_df = pd.DataFrame(index=my_index, columns=primap_years)
    historic_emissions_df.index.names = ["variable"]
    common_hist_years: list[int] = [
        int(y) for y in sorted(set(primap_years).intersection(set(years_index_gcp.tolist())))
    ]
    if not common_hist_years:
        raise ValueError(
            "No overlapping historical years between PRIMAP and GCP datasets. "
            "Cannot build harmonization baseline."
        )

    def mask_primap(dict_d: dict[str, str], primap_df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=primap_df.index, dtype=bool)
        for key in dict_d:
            mask &= primap_df.index.get_level_values(key) == dict_d[key]
        return mask

    metadata_kyotoghg_m0el = {
        "area (ISO3)": "EARTH",
        "category (IPCC2006_PRIMAP)": "M.0.EL",
        "scenario (PRIMAP-hist)": "HISTTP",
        "entity": "KYOTOGHG (AR6GWP100)",
    }
    metadata_kyotoghg_mlulucf = {
        "area (ISO3)": "EARTH",
        "category (IPCC2006_PRIMAP)": "M.LULUCF",
        "scenario (PRIMAP-hist)": "HISTTP",
        "entity": "KYOTOGHG (AR6GWP100)",
    }
    metadata_kyotoghg_mag = {
        "area (ISO3)": "EARTH",
        "category (IPCC2006_PRIMAP)": "M.AG",
        "scenario (PRIMAP-hist)": "HISTTP",
        "entity": "KYOTOGHG (AR6GWP100)",
    }
    metadata_co2_m0el = metadata_kyotoghg_m0el.copy()
    metadata_co2_m0el["entity"] = "CO2"
    metadata_co2_mlulucf = metadata_kyotoghg_mlulucf.copy()
    metadata_co2_mlulucf["entity"] = "CO2"
    metadata_co2_mag = metadata_kyotoghg_mag.copy()
    metadata_co2_mag["entity"] = "CO2"
    mask_kyoto_m0el = mask_primap(metadata_kyotoghg_m0el, primap_excl_df)
    mask_kyoto_mlulucf = mask_primap(metadata_kyotoghg_mlulucf, primap_incl_df)
    mask_kyoto_mag = mask_primap(metadata_kyotoghg_mag, primap_excl_df)
    # PRIMAP hist rows are provided in kt/yr, whereas the processed AR6
    # historical baseline is stored in Mt/yr to match the pathway tables.
    historic_emissions_df.loc["Emissions|Kyoto Gases|M.0.EL"] = (
        primap_excl_df.loc[mask_kyoto_m0el].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    historic_emissions_df.loc["Emissions|Kyoto Gases|M.AG"] = (
        primap_excl_df.loc[mask_kyoto_mag].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    historic_emissions_df.loc["Emissions|Kyoto Gases|M.LULUCF"] = (
        primap_incl_df.loc[mask_kyoto_mlulucf].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    mask_co2_m0el = mask_primap(metadata_co2_m0el, primap_excl_df)
    mask_co2_mlulucf = mask_primap(metadata_co2_mlulucf, primap_incl_df)
    mask_co2_mag = mask_primap(metadata_co2_mag, primap_excl_df)
    historic_emissions_df.loc["Emissions|CO2|M.0.EL"] = (
        primap_excl_df.loc[mask_co2_m0el].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    historic_emissions_df.loc["Emissions|CO2|M.AG"] = (
        primap_excl_df.loc[mask_co2_mag].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    historic_emissions_df.loc["Emissions|CO2|M.LULUCF"] = (
        primap_incl_df.loc[mask_co2_mlulucf].sum(axis=0).to_numpy(dtype=float) * PRIMAP_KT_TO_MT
    )
    historic_emissions_df.loc[NET_KYOTO_WO_AFOLU] = (
        historic_emissions_df.loc["Emissions|Kyoto Gases|M.0.EL"]
        - historic_emissions_df.loc["Emissions|Kyoto Gases|M.AG"]
    ).to_numpy(dtype=float)
    historic_emissions_df.loc[NET_KYOTO_WITH_AFOLU] = (
        historic_emissions_df.loc["Emissions|Kyoto Gases|M.0.EL"]
        + historic_emissions_df.loc["Emissions|Kyoto Gases|M.LULUCF"]
    ).to_numpy(dtype=float)
    historic_emissions_df.loc[NET_CO2_WO_AFOLU] = (
        historic_emissions_df.loc["Emissions|CO2|M.0.EL"]
        - historic_emissions_df.loc["Emissions|CO2|M.AG"]
    ).to_numpy(dtype=float)
    historic_emissions_df.loc[NET_CO2_WITH_AFOLU] = (
        historic_emissions_df.loc["Emissions|CO2|M.0.EL"]
        + historic_emissions_df.loc["Emissions|CO2|M.LULUCF"]
    ).to_numpy(dtype=float)
    bunkers_hist = (
        bunkers_co2_gcp_df.loc["Emissions|CO2|Bunkers"].loc[common_hist_years].to_numpy(dtype=float)
    )
    _add_year_values(
        historic_emissions_df,
        row_label=NET_KYOTO_WO_AFOLU,
        years=common_hist_years,
        values=bunkers_hist,
    )
    _add_year_values(
        historic_emissions_df,
        row_label=NET_KYOTO_WITH_AFOLU,
        years=common_hist_years,
        values=bunkers_hist,
    )
    _add_year_values(
        historic_emissions_df,
        row_label=NET_CO2_WO_AFOLU,
        years=common_hist_years,
        values=bunkers_hist,
    )
    _add_year_values(
        historic_emissions_df,
        row_label=NET_CO2_WITH_AFOLU,
        years=common_hist_years,
        values=bunkers_hist,
    )
    _set_year_values(
        historic_emissions_df,
        row_label="Emissions|CO2|Bunkers",
        years=common_hist_years,
        values=bunkers_hist,
    )
    gross_equal_net = {
        GROSS_KYOTO_WITH_AFOLU: NET_KYOTO_WITH_AFOLU,
        GROSS_KYOTO_WO_AFOLU: NET_KYOTO_WO_AFOLU,
        GROSS_CO2_WITH_AFOLU: NET_CO2_WITH_AFOLU,
        GROSS_CO2_WO_AFOLU: NET_CO2_WO_AFOLU,
        GROSS_ALT_KYOTO_WITH_AFOLU: NET_KYOTO_WITH_AFOLU,
        GROSS_ALT_KYOTO_WO_AFOLU: NET_KYOTO_WO_AFOLU,
        GROSS_ALT_CO2_WITH_AFOLU: NET_CO2_WITH_AFOLU,
        GROSS_ALT_CO2_WO_AFOLU: NET_CO2_WO_AFOLU,
    }
    for gross_variable, net_variable in gross_equal_net.items():
        historic_emissions_df.loc[gross_variable] = historic_emissions_df.loc[net_variable]
    historic_emissions_df.insert(0, column="units", value=[""] * len(historic_emissions_df))
    for idx in historic_emissions_df.index:
        historic_emissions_df.loc[idx, "units"] = "MtCO2eq/yr" if "Kyoto" in idx else "MtCO2/yr"
    latest_common_hist_year = int(common_hist_years[-1])
    historic_emissions_df.drop(
        columns=[
            col
            for col in historic_emissions_df.columns
            if str(col).isdigit() and int(col) > latest_common_hist_year
        ],
        inplace=True,
    )
    return historic_emissions_df
