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


def test_no_s00_documented_gaps_remain_in_current_source_data(result):
    # Los gaps extraordinarios que motivaron s00_gap_M23 y s00_gap_H25_M25
    # (documentados en 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md SS4.1-bis)
    # ya no existen en los archivos fuente vigentes (data/00_source fue
    # actualizada; ver manifests/s00_source_manifest.csv). config/intraday_config.yaml
    # ya no declara esos gap_id en known_gaps, asi que ninguna fecha del
    # audit debe referenciarlos.
    assert result.audit_df["s00_gap_reference"].notna().sum() == 0
    assert not (result.audit_df["day_status"] == prep.DAY_STATUS_NO_DATA_GAP_S00).any()
    assert not (result.audit_df["day_status"] == prep.DAY_STATUS_PARTIAL_GAP_S00).any()


def test_regime_distribution_sums_to_total_rows(result):
    assert result.regime_dist["n_bars"].sum() == len(result.df_window)


def test_dst_transition_days_have_no_duplicate_minutes(result):
    assert result.manifest["dst_check"]["dates_with_duplicates"] == {}


def test_full_coverage_days_are_eligible(result):
    full_days = result.audit_df[result.audit_df["day_status"] == prep.DAY_STATUS_FULL]
    assert len(full_days) > 0
    assert (full_days["is_model_eligible"]).all()
    assert (full_days["eligibility_category"] == prep.ELIGIBILITY_FULL).all()


def test_exactly_one_contract_per_date_in_final_dataset(result):
    counts = result.df_window.groupby("date")["contract"].nunique()
    assert counts.max() == 1
    assert (result.audit_df.loc[result.audit_df["observed_bars"] > 0, "n_contracts_observed"] <= 1).all()


def test_no_duplicate_timestamps_in_final_dataset(result):
    reread = pd.read_parquet(result.parquet_path)
    key = list(zip(reread["date"], reread["minute_of_day"]))
    assert len(key) == len(set(key))


def test_full_coverage_days_have_691_consecutive_minutes(result):
    full_days = result.audit_df[result.audit_df["day_status"] == prep.DAY_STATUS_FULL]
    assert (full_days["observed_bars"] == 691).all()
    assert (full_days["internal_gap_count"] == 0).all()
    assert (full_days["consecutive_segment_count"] == 1).all()
    assert (full_days["longest_consecutive_run"] == 691).all()


def test_early_close_eligible_days_have_verified_511_pattern(result):
    early = result.audit_df[result.audit_df["day_status"] == prep.DAY_STATUS_PARTIAL_EARLY_CLOSE]
    assert len(early) > 0
    assert (early["observed_bars"] == 511).all()
    assert (early["internal_gap_count"] == 0).all()
    assert (early["eligibility_category"] == prep.ELIGIBILITY_EARLY_CLOSE).all()
    assert (early["is_model_eligible"]).all()
    # No forman parte de la poblacion full_day_eligible por defecto.
    assert not (early["eligibility_category"] == prep.ELIGIBILITY_FULL).any()


# ---------------------------------------------------------------------------
# Resolucion de rollover: regresion sobre las 3 transiciones reales con
# solapamiento en los datos vigentes y los 2 casos limite documentados.
# ---------------------------------------------------------------------------

def test_rollover_regression_z24_to_h25(result):
    events = result.rollover_events
    row = events[events["outgoing_contract"] == "Z24"].iloc[0]
    assert str(row["signal_date"]) == "2024-12-17"
    assert row["incoming_contract"] == "H25"
    assert str(row["effective_date"]) == "2024-12-18"
    assert row["incoming_share"] >= 0.55


def test_rollover_regression_h25_to_m25(result):
    events = result.rollover_events
    row = events[events["outgoing_contract"] == "H25"].iloc[0]
    assert str(row["signal_date"]) == "2025-03-18"
    assert row["incoming_contract"] == "M25"
    assert str(row["effective_date"]) == "2025-03-19"
    assert row["incoming_share"] >= 0.55


def test_rollover_regression_m26_to_u26(result):
    events = result.rollover_events
    row = events[events["outgoing_contract"] == "M26"].iloc[0]
    assert str(row["signal_date"]) == "2026-06-15"
    assert row["incoming_contract"] == "U26"
    assert str(row["effective_date"]) == "2026-06-16"
    assert row["incoming_share"] >= 0.55


def test_rollover_2025_03_15_does_not_confirm(result):
    amb = result.rollover_ambiguous
    row = amb[amb["date"].astype(str) == "2025-03-15"]
    assert len(row) == 1
    assert bool(row.iloc[0]["confirmed_here"]) is False


def test_rollover_regression_2025_03_17_no_data_fallback_becomes_full_coverage(result):
    # Regla 11 (respaldo): H25 (activo) tiene 0 barras el 2025-03-17, pero
    # M25 (entrante) tiene una sesion completa real -- S01 debe conservar
    # M25 esa fecha y clasificarla full_coverage/full_day_eligible. El
    # rollover formal H25->M25 sigue confirmando el 2025-03-18 (efectivo
    # 2025-03-19), sin adelantarse por este respaldo.
    d = pd.Timestamp("2025-03-17").date()
    sub = result.df_window[result.df_window["date"] == d]
    assert sub["contract"].unique().tolist() == ["M25"]
    assert len(sub) == 691

    audit_row = result.audit_df[result.audit_df["date"] == d].iloc[0]
    assert audit_row["day_status"] == prep.DAY_STATUS_FULL
    assert audit_row["eligibility_category"] == prep.ELIGIBILITY_FULL
    assert audit_row["is_model_eligible"]

    fallback = result.rollover_ambiguous[result.rollover_ambiguous["date"] == d]
    assert len(fallback) == 1
    assert fallback.iloc[0]["selection_reason"] == "active_contract_no_data_fallback_to_incoming"
    assert fallback.iloc[0]["selected_contract_for_date"] == "M25"

    # El cruce formal no se adelanto: sigue siendo el 2025-03-18/19.
    row = result.rollover_events[result.rollover_events["outgoing_contract"] == "H25"].iloc[0]
    assert str(row["signal_date"]) == "2025-03-18"
    assert str(row["effective_date"]) == "2025-03-19"


def test_rollover_2026_06_11_keeps_m26_with_real_coverage(result):
    sub = result.df_window[result.df_window["date"] == pd.Timestamp("2026-06-11").date()]
    assert sub["contract"].unique().tolist() == ["M26"]
    audit_row = result.audit_df[result.audit_df["date"] == pd.Timestamp("2026-06-11").date()].iloc[0]
    assert audit_row["observed_bars"] == len(sub)
    assert audit_row["observed_bars"] < 691


def test_rollover_conservation_identity_is_enforced_and_reported(result):
    # Regla bloqueante: filas_pre_rollover == filas_resueltas + filas_descartadas.
    # build_manifest hace fallar la construccion si esto no se cumple (ver
    # resolve_rollovers y build_manifest en src/data/s01_intraday_preparation.py);
    # esta prueba verifica que el resultado quede ademas reportado en el manifest.
    rollover_manifest = result.manifest["rollover"]
    assert rollover_manifest["conservation_check_passed"] is True
    assert (
        rollover_manifest["n_rows_pre_rollover"]
        == rollover_manifest["n_rows_resolved"] + rollover_manifest["n_discarded_rows"]
    )
    assert rollover_manifest["n_rows_resolved"] == len(result.df_window)
    discarded = pd.read_parquet(result.rollover_discarded_path)
    assert rollover_manifest["n_discarded_rows"] == len(discarded)


def test_rollover_no_contract_ever_reverts_to_an_earlier_rank(result):
    # Regla 6 (irreversibilidad): el CONTRATO ACTIVO formal nunca retrocede.
    # Las fechas resueltas por la regla 11 (respaldo,
    # active_contract_no_data_fallback_to_incoming) son una excepcion
    # deliberada y documentada: pueden mostrar temporalmente el contrato
    # entrante ANTES de que el cruce se confirme formalmente (p.ej.
    # 2025-03-17 muestra M25 aunque el activo formal sigue siendo H25 hasta
    # el 2025-03-18/19) -- eso no es una reversion del ESTADO activo, es la
    # resolucion puntual de un dia sin datos del activo. Excluyendo esas
    # fechas, el rango de contratos por fecha debe ser monotono no
    # decreciente.
    contract_rank = {
        c: i for i, c in enumerate(
            result.df_window.groupby("contract")["date"].min().sort_values().index
        )
    }
    fallback_dates = set(
        result.rollover_ambiguous.loc[
            result.rollover_ambiguous["selection_reason"] == "active_contract_no_data_fallback_to_incoming",
            "date",
        ]
    )
    df = result.df_window[~result.df_window["date"].isin(fallback_dates)].sort_values("date")
    ranks_in_order = df["contract"].map(contract_rank).tolist()
    assert ranks_in_order == sorted(ranks_in_order), (
        "el rango de contrato retrocedio en una fecha que NO es de respaldo "
        "(violacion real de irreversibilidad, regla 6)"
    )


def test_rollover_discarded_rows_are_traced_and_disjoint_from_resolved(result):
    discarded = pd.read_parquet(result.rollover_discarded_path)
    assert len(discarded) > 0
    assert discarded["discard_reason"].notna().all()
    assert len(discarded) + len(result.df_window) >= len(result.df_window)  # sanity: no negative
    # Ninguna fila descartada corresponde a (date, contract) elegido ese dia.
    chosen_by_date = result.df_window.groupby("date")["contract"].first().to_dict()
    bad = discarded[discarded.apply(lambda r: chosen_by_date.get(r["date"]) == r["contract"], axis=1)]
    assert bad.empty


def test_rollover_ambiguous_dates_all_resolved_to_single_contract(result):
    ambiguous_dates = result.rollover_ambiguous["date"].unique()
    for d in ambiguous_dates:
        sub = result.df_window[result.df_window["date"] == d]
        assert sub["contract"].nunique() <= 1, f"{d} quedo con mas de un contrato"


def test_segment_summary_matches_consecutive_segment_ids(result):
    segs = pd.read_parquet(result.segment_summary_path)
    assert len(segs) == result.df_window.groupby(["date", "consecutive_segment_id"]).ngroups
    full_day_segs = segs[segs["length"] == 691]
    assert (full_day_segs["start_minute"] == 270).all()
    assert (full_day_segs["end_minute"] == 960).all()


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
