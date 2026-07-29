"""Pruebas de integracion de S01 v2 sobre data/01_raw/mnq_raw_v2.parquet real.

Solo lectura sobre las entradas; toda escritura de artefactos usa tmp_path,
nunca data/02_intraday/ productivo.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import s01_intraday_preparation as prep

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "intraday_config.yaml"
PRODUCTIVE_INTRADAY_DIR = PROJECT_ROOT / "data" / "02_intraday"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("s01_integration_output")
    return prep.run_s01_preparation(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        force_rebuild=True,
    )


def test_never_writes_to_productive_intraday_dir_besides_preexisting(result):
    assert PRODUCTIVE_INTRADAY_DIR not in result.parquet_path.parents


def test_artifacts_written(result):
    assert result.parquet_path.exists()
    assert result.audit_path.exists()
    assert result.regime_dist_path.exists()
    assert result.manifest_path.exists()
    assert result.summary_path.exists()
    assert result.tz_validation_path.exists()


def test_timezone_hypothesis_utc_wins(result):
    top = result.tz_table.iloc[0]
    assert top["timezone_candidate"] == "UTC"
    assert result.manifest["timezone"]["timezone_selected"] == "UTC"
    assert result.manifest["timezone"]["timezone_provider_confirmation"] is False


def test_window_minute_bounds(result):
    mod = result.df_window["minute_of_day"]
    assert mod.min() == 270
    assert mod.max() == 960


def test_schema_matches_expected(result):
    assert list(result.df_window.columns) == [
        "date", "minute_of_day", "regime_id", "regime_label", "consecutive_segment_id",
        "open", "high", "low", "close", "volume", "contract",
    ]


def test_regime_boundary_16_00_is_closing(result):
    at_close = result.df_window[result.df_window["minute_of_day"] == 960]
    assert not at_close.empty
    assert (at_close["regime_id"] == 4).all()
    assert (at_close["regime_label"] == "Closing").all()


def test_regime_boundary_04_30_is_early_premarket(result):
    at_open = result.df_window[result.df_window["minute_of_day"] == 270]
    assert not at_open.empty
    assert (at_open["regime_id"] == 0).all()
    assert (at_open["regime_label"] == "Early_Premarket").all()


def test_no_row_falls_outside_any_regime(result):
    assert result.df_window["regime_id"].isin([0, 1, 2, 3, 4]).all()


def test_trading_day_audit_covers_full_calendar_range(result):
    # El audit cubre el rango calendario COMPLETO de la fuente convertida
    # (antes del filtro de ventana), no solo las fechas con barras dentro de
    # 04:30-16:00 -- por eso se compara contra el propio rango de fechas del
    # audit, no contra df_window (que ya esta recortado a la ventana).
    dates = pd.to_datetime(result.audit_df["date"])
    expected_days = (dates.max() - dates.min()).days + 1
    assert len(result.audit_df) == expected_days
    assert result.audit_df["date"].is_unique
    # y cubre al menos todo el rango de fechas presente en el dataset filtrado
    assert dates.min().date() <= result.df_window.index.min().date()
    assert dates.max().date() >= result.df_window.index.max().date()


def test_gap_m23_dates_marked_not_eligible(result):
    sub = result.audit_df[result.audit_df["s00_gap_reference"] == "s00_gap_M23"]
    assert len(sub) > 0
    assert (~sub["is_model_eligible"]).all()
    assert (sub["day_status"].isin([
        prep.DAY_STATUS_NO_DATA_GAP_S00, prep.DAY_STATUS_PARTIAL_GAP_S00,
    ])).all()


def test_gap_h25_m25_dates_marked_not_eligible(result):
    sub = result.audit_df[result.audit_df["s00_gap_reference"] == "s00_gap_H25_M25"]
    assert len(sub) > 0
    assert (~sub["is_model_eligible"]).all()


def test_regime_distribution_sums_to_total_rows(result):
    assert result.regime_dist["n_bars"].sum() == len(result.df_window)


def test_dst_transition_days_have_no_duplicate_minutes(result):
    assert result.manifest["dst_check"]["dates_with_duplicates"] == {}


def test_full_coverage_days_are_eligible(result):
    full_days = result.audit_df[result.audit_df["day_status"] == prep.DAY_STATUS_FULL]
    assert len(full_days) > 0
    assert (full_days["is_model_eligible"]).all()
    assert (full_days["eligibility_category"] == prep.ELIGIBILITY_FULL).all()


def test_atomic_write_verified_and_rereadable(result):
    reread = pd.read_parquet(result.parquet_path)
    assert reread.shape == result.df_window.shape
    assert result.parquet_sha256 and len(result.parquet_sha256) == 64


def test_second_run_reuses_when_nothing_changed(tmp_path):
    output_dir = tmp_path / "s01_reuse_check"
    first = prep.run_s01_preparation(
        project_root=PROJECT_ROOT, config_path=CONFIG_PATH,
        output_dir=output_dir, force_rebuild=False,
    )
    second = prep.run_s01_preparation(
        project_root=PROJECT_ROOT, config_path=CONFIG_PATH,
        output_dir=output_dir, force_rebuild=False,
    )
    assert first.reused_existing is False
    assert second.reused_existing is True
    assert second.parquet_sha256 == first.parquet_sha256
