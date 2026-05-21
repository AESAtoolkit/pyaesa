import json
import importlib
from pathlib import Path
from typing import Sequence
import zipfile

import pandas as pd
import pytest
import requests

from pyaesa.download.pop_gdp import download_imf_twn as imf_mod
from pyaesa.download.pop_gdp import metadata as metadata_mod
from pyaesa.download.pop_gdp import download_ssp as ssp_mod
from pyaesa.download.pop_gdp import download_wb as wb_mod
from pyaesa.download.pop_gdp.contracts import PAST_YEAR_MIN

entry_mod = importlib.import_module("pyaesa.download.pop_gdp.download_pop_gdp")


class _JsonResponse:
    def __init__(self, payload, *, status_exception=None) -> None:
        self._payload = payload
        self._status_exception = status_exception
        try:
            self.text = json.dumps(payload)
        except TypeError:
            self.text = "{}"

    def raise_for_status(self) -> None:
        if self._status_exception is not None:
            raise self._status_exception

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _BinaryResponse:
    def __init__(self, payload: bytes, *, status_exception=None) -> None:
        self._payload = payload
        self._status_exception = status_exception

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        if self._status_exception is not None:
            raise self._status_exception

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._payload), chunk_size):
            yield self._payload[index : index + chunk_size]


def _write_wdi_zip(
    path: Path,
    *,
    country_frame: pd.DataFrame,
    data_frame: pd.DataFrame,
    include_country_member: bool = True,
    include_data_member: bool = True,
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        if include_country_member:
            archive.writestr("WDICountry.csv", country_frame.to_csv(index=False))
        if include_data_member:
            archive.writestr("WDICSV.csv", data_frame.to_csv(index=False))
    return path


def _write_cached_historical_source(output_filename: str) -> Path:
    output_path = wb_mod._get_output_path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "wb_full_name": ["Country A", "Country A"],
            "iso3_code": ["AAA", "AAA"],
            "variable": [wb_mod.POP_WB_INDICATOR, wb_mod.GDP_WB_INDICATOR],
            "unit": [wb_mod.POP_WB_UNIT, wb_mod.GDP_WB_UNIT],
            str(PAST_YEAR_MIN): [1.0, 2.0],
            "2021": [3.0, 4.0],
        }
    ).to_csv(output_path, index=False)
    metadata_mod._write_meta(
        output_filename,
        PAST_YEAR_MIN,
        2021,
        [wb_mod.POP_WB_INDICATOR, wb_mod.GDP_WB_INDICATOR],
    )
    return output_path


def _write_cached_ssp_source() -> Path:
    output_path = ssp_mod._get_output_path(ssp_mod._OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_year = int(min(ssp_mod.FUTURE_YEARS))
    last_year = int(max(ssp_mod.FUTURE_YEARS))
    pd.DataFrame(
        {
            "model": ["M1", "M1"],
            "ssp_scenario": ["SSP2", "SSP2"],
            "ssp_full_name": ["World", "World"],
            "variable": list(ssp_mod.DEFAULT_SSP_INDICATORS),
            "unit": ["u", "u"],
            str(first_year): [1.0, 2.0],
            str(last_year): [3.0, 4.0],
        }
    ).to_csv(output_path, index=False)
    metadata_mod._write_meta(
        ssp_mod._OUTPUT_FILENAME,
        first_year,
        last_year,
        list(ssp_mod.DEFAULT_SSP_INDICATORS),
    )
    return output_path


def test_download_pop_gdp_entrypoint_reuses_selected_cached_sources(
    project_repo: Path,
) -> None:
    del project_repo
    _write_cached_historical_source("wb")
    _write_cached_historical_source(imf_mod.OUTPUT_FILENAME)
    ssp_path = ssp_mod._get_output_path(ssp_mod._OUTPUT_FILENAME)
    ssp_path.unlink(missing_ok=True)

    entry_mod.download_pop_gdp(past_years=True, future_years=False, refresh=False)
    assert ssp_path.exists() is False

    _write_cached_ssp_source()
    entry_mod.download_pop_gdp(past_years=False, future_years=True, refresh=False)
    entry_mod.download_pop_gdp(past_years=True, future_years=True, refresh=False)
    entry_mod.download_pop_gdp(past_years=False, future_years=False, refresh=False)


def test_download_pop_gdp_metadata_contracts_cover_missing_and_variable_scope(
    project_repo: Path,
) -> None:
    del project_repo
    assert metadata_mod._read_meta("wb_raw") is None

    metadata_mod._write_meta("wb_raw", 2000, 2005, ["pop", "gdp", "pop"])
    payload = metadata_mod._read_meta("wb_raw")
    assert payload is not None
    assert payload["begin_year"] == 2000
    assert payload["end_year"] == 2005
    assert payload["variables"] == ["pop", "gdp", "pop"]
    assert metadata_mod._meta_covers(payload, 2001, 2004, ["pop"]) is True
    assert metadata_mod._meta_covers(payload, 1999, 2004, ["pop"]) is False
    assert metadata_mod._meta_covers(payload, 2001, 2006, ["pop"]) is False
    assert metadata_mod._meta_covers(payload, 2001, 2004, ["missing"]) is False

    with pytest.raises(RuntimeError):
        wb_mod.resolve_historical_years_from_frame(pd.DataFrame({"label": ["x"]}))


def test_world_bank_retry_contracts_cover_retry_success_and_exhaustion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    retry_attempts = {"count": 0}

    def flaky_request() -> str:
        retry_attempts["count"] += 1
        if retry_attempts["count"] < 3:
            raise requests.RequestException("temporary disconnect")
        return "ok"

    assert (
        wb_mod._run_wb_request_with_retry(
            flaky_request,
            operation="indicator 'POP'",
            max_attempts=3,
            retry_delay_seconds=0,
        )
        == "ok"
    )
    assert retry_attempts["count"] == 3
    stdout = capsys.readouterr().out
    assert stdout.strip()

    def always_fail() -> str:
        raise requests.RequestException("still down")

    with pytest.raises(RuntimeError):
        wb_mod._run_wb_request_with_retry(
            always_fail,
            operation="economy list",
            max_attempts=2,
            retry_delay_seconds=0,
        )

    never_called = {"count": 0}

    def empty_retry_loader() -> str:
        never_called["count"] += 1
        return "unexpected"

    with pytest.raises(RuntimeError):
        wb_mod._run_wb_request_with_retry(
            empty_retry_loader,
            operation="empty retry",
            max_attempts=0,
            retry_delay_seconds=0,
        )
    assert never_called["count"] == 0


def test_world_bank_bulk_download_contracts_cover_retry_and_member_validation(
    tmp_path: Path,
) -> None:
    good_zip = _write_wdi_zip(
        tmp_path / "good.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Short Name": ["Country A"],
                "Region": ["Region"],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Indicator Code": [wb_mod.POP_WB_CODE],
                str(PAST_YEAR_MIN): [1.0],
                "2026": [2.0],
            }
        ),
    ).read_bytes()
    latest_wdi_url = "https://example.test/WDI_CSV.zip"
    metadata_payload = json.dumps({"distribution": {"url": latest_wdi_url}}).encode("utf-8")
    attempts = {"count": 0}

    def flaky_get(url: str, *_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise requests.RequestException("temporary disconnect")
        if url == wb_mod._WDI_BULK_URL:
            return _BinaryResponse(metadata_payload)
        assert url == latest_wdi_url
        return _BinaryResponse(good_zip)

    target = wb_mod._download_wdi_bulk_zip(tmp_path / "downloaded.zip", request_get=flaky_get)
    assert target.exists()
    assert attempts["count"] == 3

    class _BinaryResponseWithEmptyChunk(_BinaryResponse):
        def iter_content(self, chunk_size: int):
            yield b""
            yield from super().iter_content(chunk_size)

    def streamed_get(url: str, *_args, **_kwargs):
        if url == wb_mod._WDI_BULK_URL:
            return _BinaryResponse(metadata_payload)
        assert url == latest_wdi_url
        return _BinaryResponseWithEmptyChunk(good_zip)

    streamed_target = wb_mod._download_wdi_bulk_zip(
        tmp_path / "downloaded_streamed.zip",
        request_get=streamed_get,
    )
    assert streamed_target.exists()

    bad_zip = _write_wdi_zip(
        tmp_path / "bad.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Short Name": ["Country A"],
                "Region": ["Region"],
            }
        ),
        data_frame=pd.DataFrame(),
        include_country_member=False,
    ).read_bytes()

    def bad_zip_get(url: str, *_args, **_kwargs):
        if url == wb_mod._WDI_BULK_URL:
            return _BinaryResponse(metadata_payload)
        assert url == latest_wdi_url
        return _BinaryResponse(bad_zip)

    with pytest.raises(RuntimeError):
        wb_mod._download_wdi_bulk_zip(
            tmp_path / "missing_member.zip",
            request_get=bad_zip_get,
        )


def test_world_bank_bulk_contracts_and_generation(project_repo: Path, tmp_path: Path) -> None:
    df = pd.DataFrame({"iso3_code": ["AAA"], "series_code": ["X"], "2000": [1.0]})
    out = wb_mod._ensure_year_columns(df, [2000, 2001])
    assert "2001" in out.columns

    assert metadata_mod._meta_covers({}, 1995, 2024, [wb_mod.POP_WB_INDICATOR]) is False
    empty_trimmed, empty_years = wb_mod._drop_trailing_empty_historical_years(
        pd.DataFrame(
            columns=[
                "wb_full_name",
                "iso3_code",
                "variable",
                "unit",
                str(PAST_YEAR_MIN),
                "2024",
            ]
        )
    )
    assert empty_trimmed.empty is True
    assert empty_years == [PAST_YEAR_MIN, 2024]
    with pytest.raises(RuntimeError):
        wb_mod._drop_trailing_empty_historical_years(
            pd.DataFrame(
                {
                    "wb_full_name": ["Country A"],
                    "iso3_code": ["AAA"],
                    "variable": [wb_mod.POP_WB_INDICATOR],
                    "unit": [wb_mod.POP_WB_UNIT],
                    str(PAST_YEAR_MIN): [pd.NA],
                    "2024": [pd.NA],
                }
            )
        )

    years = [PAST_YEAR_MIN, 2026]
    year_payload = {str(year): [float(year), float(year) * 10, float(year) * 100] for year in years}
    bulk_zip = _write_wdi_zip(
        tmp_path / "wdi.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA", "AFE"],
                "Short Name": ["Country A", "Aggregate"],
                "Region": ["Region", pd.NA],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["AAA", "AAA", "AFE"],
                "Indicator Code": [
                    wb_mod.POP_WB_CODE,
                    wb_mod.GDP_WB_CODE,
                    wb_mod.POP_WB_CODE,
                ],
                **year_payload,
            }
        ),
    )

    loaded = wb_mod._load_wdi_bulk_frame_from_archive(
        bulk_zip,
        indicators=wb_mod.DEFAULT_WB_CODES,
    )
    assert loaded["iso3_code"].tolist() == ["AAA", "AAA"]
    assert loaded["wb_full_name"].tolist() == ["Country A", "Country A"]
    assert set(loaded["variable"].tolist()) == {wb_mod.POP_WB_INDICATOR, wb_mod.GDP_WB_INDICATOR}
    assert set(loaded["unit"].tolist()) == {wb_mod.POP_WB_UNIT, wb_mod.GDP_WB_UNIT}
    assert "2026" in loaded.columns

    trailing_empty_zip = _write_wdi_zip(
        tmp_path / "wdi_trailing_empty.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA", "AFE"],
                "Short Name": ["Country A", "Aggregate"],
                "Region": ["Region", pd.NA],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["AAA", "AAA", "AFE"],
                "Indicator Code": [
                    wb_mod.POP_WB_CODE,
                    wb_mod.GDP_WB_CODE,
                    wb_mod.POP_WB_CODE,
                ],
                str(PAST_YEAR_MIN): [1.0, 10.0, 100.0],
                "2024": [2.0, 20.0, 200.0],
                "2025": [pd.NA, pd.NA, pd.NA],
            }
        ),
    )
    trimmed_loaded = wb_mod._load_wdi_bulk_frame_from_archive(
        trailing_empty_zip,
        indicators=wb_mod.DEFAULT_WB_CODES,
    )
    assert "2024" in trimmed_loaded.columns
    assert "2025" not in trimmed_loaded.columns

    no_real_country_zip = _write_wdi_zip(
        tmp_path / "no_real_country.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AFE"],
                "Short Name": ["Aggregate"],
                "Region": [pd.NA],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["AFE"],
                "Indicator Code": [wb_mod.POP_WB_CODE],
                "2000": [1.0],
            }
        ),
    )
    with zipfile.ZipFile(no_real_country_zip) as archive:
        with pytest.raises(RuntimeError):
            wb_mod._load_wdi_country_frame(archive)

    no_indicator_zip = _write_wdi_zip(
        tmp_path / "no_indicator.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Short Name": ["Country A"],
                "Region": ["Region"],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Indicator Code": ["OTHER"],
                "2000": [1.0],
            }
        ),
    )
    with zipfile.ZipFile(no_indicator_zip) as archive:
        with pytest.raises(RuntimeError):
            wb_mod._load_wdi_indicator_frame(
                archive,
                indicators=wb_mod.DEFAULT_WB_CODES,
            )

    no_merge_zip = _write_wdi_zip(
        tmp_path / "no_merge.zip",
        country_frame=pd.DataFrame(
            {
                "Country Code": ["AAA"],
                "Short Name": ["Country A"],
                "Region": ["Region"],
            }
        ),
        data_frame=pd.DataFrame(
            {
                "Country Code": ["BBB"],
                "Indicator Code": [wb_mod.POP_WB_CODE],
                "2000": [1.0],
            }
        ),
    )
    with pytest.raises(RuntimeError):
        wb_mod._load_wdi_bulk_frame_from_archive(
            no_merge_zip,
            indicators=wb_mod.DEFAULT_WB_CODES,
        )

    output_path = wb_mod._get_output_path(wb_mod.OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("cached\n", encoding="utf-8")
    pd.DataFrame(
        columns=[
            "wb_full_name",
            "iso3_code",
            "variable",
            "unit",
            str(PAST_YEAR_MIN),
            "2026",
        ]
    ).to_csv(
        output_path,
        index=False,
    )
    metadata_mod._write_meta(
        "wb",
        PAST_YEAR_MIN,
        2026,
        [wb_mod.POP_WB_INDICATOR, wb_mod.GDP_WB_INDICATOR],
    )
    assert wb_mod._generate_wb_raw(refresh=False) == output_path

    output_path.unlink()
    metadata_mod._get_meta_path(wb_mod.OUTPUT_FILENAME).write_text(
        "{invalid",
        encoding="utf-8",
    )

    def copy_bulk_zip(target: Path) -> Path:
        target.write_bytes(bulk_zip.read_bytes())
        return target

    result = wb_mod._generate_wb_raw(refresh=True, bulk_archive_downloader=copy_bulk_zip)
    written = pd.read_csv(result)
    assert written["iso3_code"].tolist() == ["AAA", "AAA"]
    assert set(written["variable"].tolist()) == {wb_mod.POP_WB_INDICATOR, wb_mod.GDP_WB_INDICATOR}
    assert set(written["unit"].tolist()) == {wb_mod.POP_WB_UNIT, wb_mod.GDP_WB_UNIT}
    assert "2026" in written.columns

    output_path.write_text("stale\n", encoding="utf-8")
    pd.DataFrame(
        columns=[
            "wb_full_name",
            "iso3_code",
            "variable",
            "unit",
            str(PAST_YEAR_MIN),
            "2026",
        ]
    ).to_csv(
        output_path,
        index=False,
    )
    metadata_mod._write_meta(
        "wb",
        PAST_YEAR_MIN,
        2026,
        [wb_mod.POP_WB_INDICATOR],
    )
    refresh_calls = {"downloader": 0, "loader": 0}

    def stale_copy_bulk_zip(target: Path) -> Path:
        refresh_calls["downloader"] += 1
        target.write_bytes(bulk_zip.read_bytes())
        return target

    def tracked_bulk_loader(
        archive_path: Path,
        indicators: Sequence[str],
    ) -> pd.DataFrame:
        refresh_calls["loader"] += 1
        return wb_mod._load_wdi_bulk_frame_from_archive(
            archive_path,
            indicators=indicators,
        )

    regenerated = wb_mod._generate_wb_raw(
        refresh=False,
        bulk_archive_downloader=stale_copy_bulk_zip,
        bulk_frame_loader=tracked_bulk_loader,
    )
    assert regenerated == output_path
    assert refresh_calls == {"downloader": 1, "loader": 1}


def test_imf_download_contracts_and_generation(project_repo: Path) -> None:
    wb_output_path = imf_mod._get_output_path("wb")
    wb_output_path.unlink(missing_ok=True)
    with pytest.raises(RuntimeError):
        imf_mod._resolve_imf_historical_years_from_wb_raw()

    wb_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "wb_full_name",
            "iso3_code",
            "variable",
            "unit",
            str(PAST_YEAR_MIN),
            "2021",
        ]
    ).to_csv(
        wb_output_path,
        index=False,
    )
    assert imf_mod._resolve_imf_historical_years_from_wb_raw() == [PAST_YEAR_MIN, 2021]

    series = imf_mod._fetch_imf_series(
        "PPPGDP",
        "TWN",
        [2020, 2021],
        http_get=lambda _url, timeout=30: _JsonResponse(
            {
                "values": {
                    "PPPGDP": {
                        "TWN": {
                            "2020": 1.0,
                            "2021": 2.0,
                        }
                    }
                }
            }
        ),
    )
    assert list(series.index.astype("Int64")) == [2020, 2021]
    assert float(series.loc[2021]) == 2.0

    df = pd.DataFrame(
        {
            "gdp_ppp_usd": [90.0, 100.0, 110.0],
            "real_gdp_growth": [5.0, 10.0, 10.0],
        },
        index=[2020, 2021, 2022],
    )
    chained = imf_mod._chain_to_constant_base(
        df,
        value_col="gdp_ppp_usd",
        growth_col="real_gdp_growth",
        base_year=2021,
    )
    assert float(chained.loc[2021]) == 100.0
    assert round(float(chained.loc[2020]), 2) == round(100.0 / 1.10, 2)
    assert round(float(chained.loc[2022]), 2) == 110.0
    with pytest.raises(ValueError):
        imf_mod._chain_to_constant_base(
            df,
            value_col="gdp_ppp_usd",
            growth_col="real_gdp_growth",
            base_year=2019,
        )

    edge_df = pd.DataFrame(
        {
            "gdp_ppp_usd": [80.0, 100.0, 130.0],
            "real_gdp_growth": [5.0, 10.0, pd.NA],
        },
        index=pd.Index([2019, 2021, 2023], dtype="object"),
    )
    edge_chained = imf_mod._chain_to_constant_base(
        edge_df,
        value_col="gdp_ppp_usd",
        growth_col="real_gdp_growth",
        base_year=2021,
    )
    assert pd.isna(edge_chained.loc[2022])
    assert pd.isna(edge_chained.loc[2023])
    assert pd.isna(edge_chained.loc[2020])
    assert pd.isna(edge_chained.loc[2019])

    fetch_payload = {
        "PPPGDP": pd.Series({2021: 100.0}),
        "LP": pd.Series({2021: 20.0}),
        "NGDP_RPCH": pd.Series({2021: 5.0}),
    }
    pop_gdp = imf_mod._get_imf_twn_pop_gdp(
        [2020, 2021],
        fetch_series=lambda code, _iso3, _years: fetch_payload[code],
    )
    gdp_row = pop_gdp[pop_gdp["variable"] == imf_mod.GDP_WB_INDICATOR].iloc[0]
    pop_row = pop_gdp[pop_gdp["variable"] == imf_mod.POP_WB_INDICATOR].iloc[0]
    assert pd.isna(gdp_row["2020"])
    assert float(gdp_row["2021"]) == 100.0 * 1_000_000_000
    assert float(pop_row["2021"]) == 20.0 * 1_000_000

    output_path = imf_mod._get_output_path(imf_mod.OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("cached\n", encoding="utf-8")
    metadata_mod._write_meta(
        imf_mod.OUTPUT_FILENAME,
        2020,
        2021,
        [imf_mod.POP_WB_INDICATOR, imf_mod.GDP_WB_INDICATOR],
    )
    assert (
        imf_mod._generate_imf_twn_raw(
            refresh=False,
            historical_years_loader=lambda: [2020, 2021],
        )
        == output_path
    )

    output_path.unlink()
    metadata_mod._get_meta_path(imf_mod.OUTPUT_FILENAME).write_text(
        "{invalid",
        encoding="utf-8",
    )
    written_path = imf_mod._generate_imf_twn_raw(
        refresh=True,
        pop_gdp_loader=lambda years: pd.DataFrame(
            {
                "wb_full_name": [imf_mod.IMF_TWN_NAME, imf_mod.IMF_TWN_NAME],
                "iso3_code": [imf_mod.IMF_TWN_ISO3, imf_mod.IMF_TWN_ISO3],
                "variable": [imf_mod.POP_WB_INDICATOR, imf_mod.GDP_WB_INDICATOR],
                "unit": [imf_mod.POP_WB_UNIT, imf_mod.GDP_WB_UNIT],
                **{str(year): [1.0, 2.0] for year in years},
            }
        ),
        historical_years_loader=lambda: [2020, 2021],
    )
    written = pd.read_csv(written_path)
    assert set(written["variable"].tolist()) == {
        imf_mod.POP_WB_INDICATOR,
        imf_mod.GDP_WB_INDICATOR,
    }

    output_path.unlink()
    pop_only_path = imf_mod._generate_imf_twn_raw(
        refresh=True,
        pop_gdp_loader=lambda years: pd.DataFrame(
            {
                "wb_full_name": [imf_mod.IMF_TWN_NAME],
                "iso3_code": [imf_mod.IMF_TWN_ISO3],
                "variable": [imf_mod.POP_WB_INDICATOR],
                "unit": [imf_mod.POP_WB_UNIT],
                **{str(year): [1.0] for year in years},
            }
        ),
        historical_years_loader=lambda: [2020, 2021],
    )
    assert pd.read_csv(pop_only_path)["variable"].tolist() == [imf_mod.POP_WB_INDICATOR]

    output_path.unlink()
    gdp_only_path = imf_mod._generate_imf_twn_raw(
        refresh=True,
        pop_gdp_loader=lambda years: pd.DataFrame(
            {
                "wb_full_name": [imf_mod.IMF_TWN_NAME],
                "iso3_code": [imf_mod.IMF_TWN_ISO3],
                "variable": [imf_mod.GDP_WB_INDICATOR],
                "unit": [imf_mod.GDP_WB_UNIT],
                **{str(year): [2.0] for year in years},
            }
        ),
        historical_years_loader=lambda: [2020, 2021],
    )
    assert pd.read_csv(gdp_only_path)["variable"].tolist() == [imf_mod.GDP_WB_INDICATOR]


def test_ssp_download_contracts_and_generation(project_repo: Path) -> None:
    with pytest.raises(RuntimeError):
        ssp_mod._http_json_get(
            "https://example/api",
            http_get=lambda *_args, **_kwargs: _JsonResponse(
                {},
                status_exception=requests.RequestException("boom"),
            ),
        )

    with pytest.raises(RuntimeError):
        ssp_mod._http_json_get(
            "https://example/api",
            http_get=lambda *_args, **_kwargs: _JsonResponse(ValueError("bad")),
        )

    with pytest.raises(RuntimeError):
        ssp_mod._get_anonymous_auth_header(
            http_get=lambda *_args, **_kwargs: _JsonResponse({}),
        )

    def apps_http_get(url: str, **_kwargs):
        if url.endswith("/legacy/applications"):
            return _JsonResponse(
                [
                    {"name": "IXSE_SSP", "config": [{"path": "env", "value": "ssp"}]},
                ]
            )
        if url.endswith("/legacy/applications/IXSE_SSP/config"):
            return _JsonResponse([{"path": "baseUrl", "value": "https://ssp.example"}])
        raise AssertionError(url)

    assert (
        ssp_mod._resolve_ssp_base_url(
            {"Authorization": "Bearer x"},
            auth_url="https://auth.example",
            http_get=apps_http_get,
        )
        == "https://ssp.example"
    )

    with pytest.raises(RuntimeError):
        ssp_mod._resolve_ssp_base_url(
            {"Authorization": "Bearer x"},
            auth_url="https://auth.example",
            http_get=lambda *_args, **_kwargs: _JsonResponse({"bad": "payload"}),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._resolve_ssp_base_url(
            {"Authorization": "Bearer x"},
            auth_url="https://auth.example",
            http_get=lambda *_args, **_kwargs: _JsonResponse(
                [
                    "bad-entry",
                    {"name": "IXSE_SSP", "config": "bad"},
                    {"name": "ignored", "config": [{"path": "env", "value": "other"}]},
                    {"name": " ", "config": [{"path": "env", "value": "ssp"}]},
                ]
            ),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._resolve_ssp_base_url(
            {"Authorization": "Bearer x"},
            auth_url="https://auth.example",
            http_get=lambda url, **_kwargs: (
                _JsonResponse([{"name": "IXSE_SSP", "config": [{"path": "env", "value": "ssp"}]}])
                if url.endswith("/legacy/applications")
                else _JsonResponse([{"path": "other", "value": "missing"}])
            ),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._resolve_ssp_base_url(
            {"Authorization": "Bearer x"},
            auth_url="https://auth.example",
            http_get=lambda url, **_kwargs: (
                _JsonResponse([{"name": "IXSE_SSP", "config": [{"path": "env", "value": "ssp"}]}])
                if url.endswith("/legacy/applications")
                else _JsonResponse({"path": "baseUrl", "value": "https://ssp.example"})
            ),
        )

    with pytest.raises(RuntimeError):
        ssp_mod._fetch_default_run_ids(
            base_url="https://ssp.example",
            auth_header={},
            http_get=lambda *_args, **_kwargs: _JsonResponse([]),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._fetch_default_run_ids(
            base_url="https://ssp.example",
            auth_header={},
            http_get=lambda *_args, **_kwargs: _JsonResponse(
                [],
                status_exception=requests.RequestException("boom"),
            ),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._fetch_default_run_ids(
            base_url="https://ssp.example",
            auth_header={},
            http_get=lambda *_args, **_kwargs: _JsonResponse([{"run_id": "bad"}]),
            read_json=lambda *_args, **_kwargs: pd.DataFrame({"run_id": ["bad"]}),
        )

    run_ids = ssp_mod._fetch_default_run_ids(
        base_url="https://ssp.example",
        auth_header={},
        http_get=lambda *_args, **_kwargs: _JsonResponse(
            [{"run_id": 10}, {"run_id": 10}, {"run_id": 12}]
        ),
    )
    assert run_ids == [10, 12]

    def full_http_get(url: str, **_kwargs):
        if url.endswith("/legacy/anonym/"):
            return _JsonResponse("token")
        if url.endswith("/legacy/applications"):
            return _JsonResponse(
                [{"name": "IXSE_SSP", "config": [{"path": "env", "value": "ssp"}]}]
            )
        if url.endswith("/legacy/applications/IXSE_SSP/config"):
            return _JsonResponse([{"path": "baseUrl", "value": "https://ssp.example"}])
        if url.endswith("/runs?getOnlyDefaultRuns=true&includeMetadata=false"):
            return _JsonResponse([{"run_id": 10}])
        raise AssertionError(url)

    def full_http_post(_url: str, **_kwargs):
        return _JsonResponse(
            [
                {
                    "model": "M1",
                    "scenario": "SSP2",
                    "region": "France",
                    "variable": ssp_mod.DEFAULT_SSP_INDICATORS[0],
                    "unit": "u",
                    "year": ssp_mod.FUTURE_YEARS[0],
                    "value": 1.0,
                },
                {
                    "model": "M1",
                    "scenario": "SSP2",
                    "region": "France",
                    "variable": ssp_mod.DEFAULT_SSP_INDICATORS[1],
                    "unit": "u",
                    "year": ssp_mod.FUTURE_YEARS[0],
                    "value": 2.0,
                },
            ]
        )

    ssp_data = ssp_mod._fetch_ssp_timeseries(
        variables=ssp_mod.DEFAULT_SSP_INDICATORS,
        auth_url="https://auth.example",
        http_get=full_http_get,
        http_post=full_http_post,
    )
    assert sorted(ssp_data["variable"].unique().tolist()) == sorted(ssp_mod.DEFAULT_SSP_INDICATORS)
    assert "ssp_scenario" in ssp_data.columns
    assert "scenario" not in ssp_data.columns

    filtered_data = ssp_mod._fetch_ssp_timeseries(
        variables=["Population"],
        auth_url="https://auth.example",
        http_get=full_http_get,
        http_post=lambda _url, **_kwargs: _JsonResponse(
            [
                {
                    "model": "M1",
                    "scenario": "SSP2",
                    "region": "France",
                    "variable": "Population",
                    "unit": "u",
                    "year": 2030,
                    "value": 1.0,
                    "subannual": "Year",
                },
                {
                    "model": "M1",
                    "scenario": "SSP2",
                    "region": "France",
                    "variable": "Population",
                    "unit": "u",
                    "year": 2030,
                    "value": 9.0,
                    "subannual": "Month",
                },
            ]
        ),
    )
    assert filtered_data["value"].tolist() == [1.0]

    with pytest.raises(RuntimeError):
        ssp_mod._fetch_ssp_timeseries(
            variables=["Population"],
            auth_url="https://auth.example",
            http_get=full_http_get,
            http_post=lambda _url, **_kwargs: _JsonResponse(
                [
                    {
                        "model": "M1",
                        "scenario": "SSP2",
                        "region": "France",
                        "variable": "GDP|PPP",
                        "unit": "u",
                        "year": 2030,
                        "value": 1.0,
                    }
                ]
            ),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._fetch_ssp_timeseries(
            variables=["Population"],
            auth_url="https://auth.example",
            http_get=full_http_get,
            http_post=lambda _url, **_kwargs: _JsonResponse(
                [],
                status_exception=requests.RequestException("boom"),
            ),
        )
    with pytest.raises(RuntimeError):
        ssp_mod._fetch_ssp_timeseries(
            variables=["Population"],
            auth_url="https://auth.example",
            http_get=full_http_get,
            http_post=lambda _url, **_kwargs: _JsonResponse([{"model": "M1"}]),
        )
    with pytest.raises(RuntimeError):
        read_json_calls = iter(
            [
                pd.DataFrame({"run_id": [10]}),
                pd.DataFrame(
                    columns=["model", "scenario", "region", "variable", "unit", "year", "value"]
                ),
            ]
        )
        ssp_mod._fetch_ssp_timeseries(
            variables=["Population"],
            auth_url="https://auth.example",
            http_get=full_http_get,
            http_post=lambda _url, **_kwargs: _JsonResponse([]),
            read_json=lambda *_args, **_kwargs: next(read_json_calls),
        )

    output_path = ssp_mod._get_output_path(ssp_mod._OUTPUT_FILENAME)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("cached\n", encoding="utf-8")
    metadata_mod._write_meta(
        ssp_mod._OUTPUT_FILENAME,
        min(ssp_mod.FUTURE_YEARS),
        max(ssp_mod.FUTURE_YEARS),
        list(ssp_mod.DEFAULT_SSP_INDICATORS),
    )
    assert ssp_mod._generate_ssp_raw(refresh=False) == output_path

    output_path.unlink()
    metadata_mod._get_meta_path(ssp_mod._OUTPUT_FILENAME).write_text(
        "{invalid",
        encoding="utf-8",
    )
    fetched_path = ssp_mod._generate_ssp_raw(
        refresh=True,
        fetch_timeseries_func=lambda *, variables: pd.DataFrame(
            {
                "model": ["M1", "M1"],
                "ssp_scenario": ["SSP2", "SSP2"],
                "region": ["France", "France"],
                "variable": list(variables),
                "unit": ["u", "u"],
                "year": [ssp_mod.FUTURE_YEARS[0], ssp_mod.FUTURE_YEARS[0]],
                "value": [1.0, 2.0],
            }
        ),
    )
    assert fetched_path.exists()

    output_path.unlink()
    written_path = ssp_mod._generate_ssp_raw(
        refresh=True,
        timeseries_loader=lambda _variables: pd.DataFrame(
            {
                "model": ["M1", "M1"],
                "ssp_scenario": ["SSP2", "SSP2"],
                "region": ["France", "France"],
                "variable": list(ssp_mod.DEFAULT_SSP_INDICATORS),
                "unit": ["u", "u"],
                "year": [ssp_mod.FUTURE_YEARS[0], ssp_mod.FUTURE_YEARS[0]],
                "value": [1.0, 2.0],
            }
        ),
    )
    written = pd.read_csv(written_path)
    assert sorted(written["variable"].tolist()) == sorted(ssp_mod.DEFAULT_SSP_INDICATORS)
    assert "ssp_scenario" in written.columns
    assert "scenario" not in written.columns
