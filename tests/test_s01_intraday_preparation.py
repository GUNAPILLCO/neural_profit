"""Pruebas unitarias rapidas de S01 v2 -- datos sinteticos en memoria."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import s01_intraday_preparation as prep


def make_window_cfg():
    return {"start_time": "04:30:00", "end_time": "16:00:00", "start_minute": 270, "end_minute": 960, "expected_minutes": 691}


def make_regimes_cfg():
    return [
        {"regime_id": 0, "label": "Early_Premarket", "start_minute": 270, "end_minute": 509},
        {"regime_id": 1, "label": "Premarket", "start_minute": 510, "end_minute": 569},
        {"regime_id": 2, "label": "Opening", "start_minute": 570, "end_minute": 629},
        {"regime_id": 3, "label": "Regular", "start_minute": 630, "end_minute": 899},
        {"regime_id": 4, "label": "Closing", "start_minute": 900, "end_minute": 960},
    ]


# ---------------------------------------------------------------------------
# Regimenes: limites exactos, sin ruta default
# ---------------------------------------------------------------------------

def test_regime_lookup_covers_full_window_no_default():
    lookup, labels = prep.build_regime_lookup(make_regimes_cfg(), make_window_cfg())
    window_minutes = np.arange(270, 961)
    assert (lookup[window_minutes] != -1).all()


@pytest.mark.parametrize("minute,expected_regime", [
    (8 * 60 + 29, 0),   # 08:29 -> 0
    (8 * 60 + 30, 1),   # 08:30 -> 1
    (9 * 60 + 29, 1),   # 09:29 -> 1
    (9 * 60 + 30, 2),   # 09:30 -> 2
    (10 * 60 + 29, 2),  # 10:29 -> 2
    (10 * 60 + 30, 3),  # 10:30 -> 3
    (14 * 60 + 59, 3),  # 14:59 -> 3
    (15 * 60 + 0, 4),   # 15:00 -> 4
    (16 * 60 + 0, 4),   # 16:00 -> 4
])
def test_regime_boundary_points(minute, expected_regime):
    lookup, labels = prep.build_regime_lookup(make_regimes_cfg(), make_window_cfg())
    regime_id, regime_label = prep.assign_regime(np.array([minute]), lookup, labels)
    assert regime_id[0] == expected_regime


def test_regime_lookup_raises_on_incomplete_config():
    # Config con un hueco deliberado (falta 630-899) debe hacer fallar la
    # construccion del lookup -- ninguna ruta default silenciosa permitida.
    broken_regimes = [
        {"regime_id": 0, "label": "Early_Premarket", "start_minute": 270, "end_minute": 509},
        {"regime_id": 1, "label": "Premarket", "start_minute": 510, "end_minute": 569},
        {"regime_id": 2, "label": "Opening", "start_minute": 570, "end_minute": 629},
        {"regime_id": 4, "label": "Closing", "start_minute": 900, "end_minute": 960},
    ]
    with pytest.raises(prep.IngestionError, match="sin regimen asignado"):
        prep.build_regime_lookup(broken_regimes, make_window_cfg())


def test_assign_regime_raises_for_minute_outside_any_regime():
    lookup, labels = prep.build_regime_lookup(make_regimes_cfg(), make_window_cfg())
    # 04:29 (minuto 269) esta fuera de la ventana/regimenes definidos.
    with pytest.raises(prep.IngestionError, match="fuera de cualquier regimen"):
        prep.assign_regime(np.array([269]), lookup, labels)


# ---------------------------------------------------------------------------
# Segmentos consecutivos
# ---------------------------------------------------------------------------

def make_window_df(rows):
    """rows: list of (date, minute_of_day) -> DataFrame minimo para pruebas."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="min", tz="America/New_York")
    df = pd.DataFrame({
        "date": [r[0] for r in rows],
        "minute_of_day": [r[1] for r in rows],
        "close": range(len(rows)),
    }, index=idx)
    return df


def test_consecutive_segments_single_run():
    df = make_window_df([(date(2024, 1, 2), m) for m in range(270, 275)])
    seg = prep.assign_consecutive_segments(df)
    assert len(set(seg)) == 1


def test_consecutive_segments_splits_on_gap():
    rows = [(date(2024, 1, 2), 270), (date(2024, 1, 2), 271), (date(2024, 1, 2), 280)]
    df = make_window_df(rows)
    seg = prep.assign_consecutive_segments(df)
    assert seg[0] == seg[1]
    assert seg[2] != seg[1]


def test_consecutive_segments_resets_on_new_date():
    rows = [(date(2024, 1, 2), 959), (date(2024, 1, 2), 960), (date(2024, 1, 3), 270)]
    df = make_window_df(rows)
    seg = prep.assign_consecutive_segments(df)
    assert seg[0] == seg[1]
    assert seg[2] != seg[1]


# ---------------------------------------------------------------------------
# Auditoria de jornadas
# ---------------------------------------------------------------------------

def _full_day_df(d, window_cfg):
    minutes = list(range(window_cfg["start_minute"], window_cfg["end_minute"] + 1))
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=m) for m in minutes]
    )
    df = pd.DataFrame({
        "date": [d] * len(minutes),
        "minute_of_day": minutes,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "contract": "H20",
    }, index=idx)
    df["consecutive_segment_id"] = prep.assign_consecutive_segments(df)
    df["regime_id"] = 0
    df["regime_label"] = "x"
    return df


def test_trading_day_audit_full_coverage_day():
    window_cfg = make_window_cfg()
    d = date(2024, 1, 2)  # Tuesday, a real CME trading day
    df = _full_day_df(d, window_cfg)
    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(df, make_regimes_cfg(), window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["observed_bars"] == 691
    assert row["is_fully_consecutive"]
    assert row["day_status"] == prep.DAY_STATUS_FULL
    assert row["eligibility_category"] == prep.ELIGIBILITY_FULL
    assert row["is_model_eligible"]


def test_trading_day_audit_weekend_no_data():
    window_cfg = make_window_cfg()
    d = date(2024, 1, 6)  # Saturday
    empty_df = pd.DataFrame(columns=["date", "minute_of_day", "consecutive_segment_id", "close"])
    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(empty_df, make_regimes_cfg(), window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["observed_bars"] == 0
    assert row["calendar_status"] == "weekend"
    assert row["day_status"] == prep.DAY_STATUS_NO_DATA_WEEKEND
    assert not row["is_model_eligible"]
    assert row["eligibility_category"] == prep.ELIGIBILITY_NOT_ELIGIBLE


def test_trading_day_audit_gap_documented_s00_marks_not_eligible():
    window_cfg = make_window_cfg()
    d = date(2023, 4, 10)  # dentro del gap M23 sintetico
    empty_df = pd.DataFrame(columns=["date", "minute_of_day", "consecutive_segment_id", "close"])
    known_gaps = [{
        "gap_id": "s00_gap_M23", "gap_type": "intra_file",
        "start": "2023-04-05T18:03:00", "end": "2023-04-16T14:18:00",
    }]
    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(empty_df, make_regimes_cfg(), window_cfg, known_gaps, date_range)
    row = audit.iloc[0]
    assert row["s00_gap_reference"] == "s00_gap_M23"
    assert row["day_status"] == prep.DAY_STATUS_NO_DATA_GAP_S00
    assert not row["is_model_eligible"]
    assert row["eligibility_category"] == prep.ELIGIBILITY_NOT_ELIGIBLE


def test_trading_day_audit_partial_regime_eligible():
    window_cfg = make_window_cfg()
    d = date(2024, 1, 2)
    df = _full_day_df(d, window_cfg)
    # Elimina todo el regimen 0 (Early_Premarket, 270-509) pero deja el
    # resto intacto y consecutivo -> deberia calificar partial_regime_eligible.
    df = df[df["minute_of_day"] > 509].copy()
    df["consecutive_segment_id"] = prep.assign_consecutive_segments(df)
    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(df, make_regimes_cfg(), window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["observed_bars"] < row["expected_bars"]
    assert row["eligibility_category"] == prep.ELIGIBILITY_PARTIAL_REGIME
    assert row["is_model_eligible"]
    assert row["regime_0_observed_bars"] == 0
    assert row["regime_1_is_consecutive"]


def test_trading_day_audit_verified_early_close_511_bars():
    window_cfg = make_window_cfg()
    early_close_days = prep.get_cme_early_close_dates(date(2020, 1, 1), date(2026, 12, 31))
    assert early_close_days, "se necesita al menos una fecha real de cierre anticipado CME para esta prueba"
    d = sorted(early_close_days.keys())[0]

    minutes = list(range(270, 781))  # 04:30-13:00, 511 minutos
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=m) for m in minutes]
    )
    df = pd.DataFrame({
        "date": [d] * len(minutes), "minute_of_day": minutes,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "contract": "H20",
    }, index=idx)
    df["consecutive_segment_id"] = prep.assign_consecutive_segments(df)
    df["regime_id"] = 0
    df["regime_label"] = "x"

    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(df, make_regimes_cfg(), window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["observed_bars"] == 511
    assert row["day_status"] == prep.DAY_STATUS_PARTIAL_EARLY_CLOSE
    assert row["eligibility_category"] == prep.ELIGIBILITY_EARLY_CLOSE
    assert row["is_model_eligible"]
    assert row["eligibility_category"] != prep.ELIGIBILITY_FULL


def test_trading_day_audit_early_close_calendar_but_bars_dont_match_falls_to_undetermined():
    # El calendario CME marca cierre anticipado, pero los datos observados
    # no calzan el patron verificado (511 barras 04:30-13:00 exactas) --
    # no debe asumirse el cierre anticipado sin verificacion empirica.
    window_cfg = make_window_cfg()
    early_close_days = prep.get_cme_early_close_dates(date(2020, 1, 1), date(2026, 12, 31))
    d = sorted(early_close_days.keys())[0]

    minutes = list(range(270, 700))  # patron que NO calza 04:30-13:00 exacto
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=m) for m in minutes]
    )
    df = pd.DataFrame({
        "date": [d] * len(minutes), "minute_of_day": minutes,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "contract": "H20",
    }, index=idx)
    df["consecutive_segment_id"] = prep.assign_consecutive_segments(df)
    df["regime_id"] = 0
    df["regime_label"] = "x"

    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(df, make_regimes_cfg(), window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["day_status"] != prep.DAY_STATUS_PARTIAL_EARLY_CLOSE
    assert row["eligibility_category"] != prep.ELIGIBILITY_EARLY_CLOSE


def test_trading_day_audit_raises_if_more_than_one_contract_same_date():
    # Invariante defensiva: para cuando build_trading_day_audit recibe una
    # entrada que NO paso por resolve_rollovers (p.ej. uso incorrecto desde
    # otra etapa) -- nunca debe clasificar silenciosamente una fecha con mas
    # de un contrato.
    window_cfg = make_window_cfg()
    d = date(2024, 1, 2)
    df = _full_day_df(d, window_cfg)
    other = df.copy()
    other["contract"] = "M20"
    mixed = pd.concat([df, other])
    mixed["consecutive_segment_id"] = prep.assign_consecutive_segments(mixed)
    date_range = pd.date_range(d, d, freq="D")
    with pytest.raises(prep.IngestionError, match="mas de un contrato"):
        prep.build_trading_day_audit(mixed, make_regimes_cfg(), window_cfg, [], date_range)


def test_trading_day_audit_descriptive_only_when_no_regime_fully_covered():
    window_cfg = make_window_cfg()
    d = date(2024, 1, 2)
    df = _full_day_df(d, window_cfg)
    # Elimina un minuto en medio de CADA regimen -> ningun regimen queda
    # completo/consecutivo, pero sigue habiendo datos parciales.
    regimes = make_regimes_cfg()
    drop_minutes = [r["start_minute"] + 2 for r in regimes]
    df = df[~df["minute_of_day"].isin(drop_minutes)].copy()
    df["consecutive_segment_id"] = prep.assign_consecutive_segments(df)
    date_range = pd.date_range(d, d, freq="D")
    audit = prep.build_trading_day_audit(df, regimes, window_cfg, [], date_range)
    row = audit.iloc[0]
    assert row["eligibility_category"] == prep.ELIGIBILITY_DESCRIPTIVE
    assert not row["is_model_eligible"]


# ---------------------------------------------------------------------------
# Resolucion de rollover (datos sinteticos)
# ---------------------------------------------------------------------------

def _synthetic_full_day(d, contract, volume_per_bar=1, window_cfg=None):
    window_cfg = window_cfg or make_window_cfg()
    minutes = list(range(window_cfg["start_minute"], window_cfg["end_minute"] + 1))
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=m) for m in minutes]
    )
    return pd.DataFrame({
        "date": [d] * len(minutes),
        "minute_of_day": minutes,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": volume_per_bar, "contract": contract,
    }, index=idx)


def _synthetic_partial_day(d, contract, minutes, volume_per_bar=1):
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(minutes=m) for m in minutes]
    )
    return pd.DataFrame({
        "date": [d] * len(minutes),
        "minute_of_day": minutes,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": volume_per_bar, "contract": contract,
    }, index=idx)


def test_resolve_rollovers_clean_handoff_no_overlap_advances_without_confirmation():
    window_cfg = make_window_cfg()
    df = pd.concat([
        _synthetic_full_day(date(2024, 1, 2), "H24", window_cfg=window_cfg),
        _synthetic_full_day(date(2024, 1, 3), "M24", window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg)
    assert resolved.groupby("date")["contract"].nunique().max() == 1
    assert resolved[resolved["date"] == date(2024, 1, 3)]["contract"].unique().tolist() == ["M24"]
    assert events.empty  # sin ambiguedad real, no hay "senal" que registrar
    assert discarded.empty


def test_resolve_rollovers_confirms_on_full_session_above_threshold():
    window_cfg = make_window_cfg()
    d1, d2 = date(2024, 3, 18), date(2024, 3, 19)
    df = pd.concat([
        _synthetic_full_day(d1, "H24", volume_per_bar=1, window_cfg=window_cfg),   # 691 vol
        _synthetic_full_day(d1, "M24", volume_per_bar=2, window_cfg=window_cfg),   # 1382 vol -> share 2/3=0.667
        _synthetic_full_day(d2, "M24", volume_per_bar=1, window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["signal_date"] == d1
    assert row["outgoing_contract"] == "H24"
    assert row["incoming_contract"] == "M24"
    assert row["effective_date"] == d2
    assert row["incoming_share"] == pytest.approx(2 / 3)

    # La fecha de la senal SIGUE usando el saliente; el entrante recien
    # aplica desde la jornada siguiente.
    assert resolved[resolved["date"] == d1]["contract"].unique().tolist() == ["H24"]
    assert resolved[resolved["date"] == d2]["contract"].unique().tolist() == ["M24"]
    assert set(discarded["discard_reason"]) == {"confirmed_rollover_signal"}


def test_resolve_rollovers_does_not_confirm_below_threshold():
    window_cfg = make_window_cfg()
    d1 = date(2024, 3, 18)
    df = pd.concat([
        _synthetic_full_day(d1, "H24", volume_per_bar=10, window_cfg=window_cfg),  # share = 1/11 ~ 0.09
        _synthetic_full_day(d1, "M24", volume_per_bar=1, window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)
    assert events.empty
    assert resolved[resolved["date"] == d1]["contract"].unique().tolist() == ["H24"]
    row = ambiguous.iloc[0]
    assert bool(row["confirmed_here"]) is False
    assert bool(row["is_confirming_session"]) is True  # sesion completa, pero bajo el umbral


def test_resolve_rollovers_partial_session_never_confirms_even_above_threshold():
    window_cfg = make_window_cfg()
    d1 = date(2024, 3, 18)
    df = pd.concat([
        _synthetic_full_day(d1, "H24", volume_per_bar=1, window_cfg=window_cfg),
        _synthetic_partial_day(d1, "M24", minutes=list(range(270, 700)), volume_per_bar=100),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)
    assert events.empty  # nunca confirma: sesion no fue completa (691/691)
    row = ambiguous.iloc[0]
    assert bool(row["both_full_691_session"]) is False
    assert bool(row["confirmed_here"]) is False
    assert resolved[resolved["date"] == d1]["contract"].unique().tolist() == ["H24"]


def test_resolve_rollovers_no_data_fallback_uses_incoming_full_session():
    # Regla 11 (respaldo): el saliente tiene EXACTAMENTE 0 barras un dia
    # intermedio dentro de una ventana de solapamiento sin confirmar, pero
    # el entrante SI tiene una sesion completa -- se selecciona el
    # entrante para ESA fecha puntual (full_coverage cuando corresponda),
    # sin adelantar formalmente el contrato activo.
    window_cfg = make_window_cfg()
    d0, d_gap, d1 = date(2024, 3, 17), date(2024, 3, 18), date(2024, 3, 19)
    df = pd.concat([
        _synthetic_full_day(d0, "H24", window_cfg=window_cfg),
        _synthetic_full_day(d_gap, "M24", window_cfg=window_cfg),  # H24 ausente este dia
        _synthetic_full_day(d1, "H24", window_cfg=window_cfg),      # H24 reaparece
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg)

    assert events.empty  # el respaldo NO es una confirmacion formal de rollover
    gap_rows = resolved[resolved["date"] == d_gap]
    assert gap_rows["contract"].unique().tolist() == ["M24"]
    assert len(gap_rows) == 691  # sesion completa real del entrante -> full_coverage aguas abajo
    assert discarded[discarded["date"] == d_gap].empty  # nada se descarta: no habia nada del activo

    fallback_row = ambiguous[ambiguous["date"] == d_gap].iloc[0]
    assert fallback_row["selection_reason"] == "active_contract_no_data_fallback_to_incoming"
    assert fallback_row["selected_contract_for_date"] == "M24"
    assert bool(fallback_row["confirmed_here"]) is False

    # H24 reaparece el dia siguiente y el activo NO fue adelantado por el
    # respaldo -- sigue siendo H24 (sin confirmacion formal de volumen).
    assert resolved[resolved["date"] == d1]["contract"].unique().tolist() == ["H24"]


def test_resolve_rollovers_no_data_fallback_preserves_partial_coverage():
    # Si el entrante solo tiene cobertura parcial ese dia, se conserva tal
    # cual (sin rellenar ni completar barras sinteticas).
    window_cfg = make_window_cfg()
    d0, d_gap, d1 = date(2024, 3, 17), date(2024, 3, 18), date(2024, 3, 19)
    df = pd.concat([
        _synthetic_full_day(d0, "H24", window_cfg=window_cfg),
        _synthetic_partial_day(d_gap, "M24", minutes=[270, 271, 272]),  # H24 ausente, M24 parcial
        _synthetic_full_day(d1, "H24", window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg)
    gap_rows = resolved[resolved["date"] == d_gap]
    assert gap_rows["contract"].unique().tolist() == ["M24"]
    assert len(gap_rows) == 3  # cobertura parcial real, no completada ni descartada
    assert events.empty


def test_resolve_rollovers_regression_2025_03_17_h25_m25():
    # Caso de regresion obligatorio: H25 (activo) 0 barras el 2025-03-17,
    # M25 (entrante) 691 barras validas -- S01 debe conservar M25 esa
    # fecha, y el rollover formal H25->M25 sigue confirmando el 2025-03-18
    # (efectivo desde el 2025-03-19), sin adelantarse por el respaldo.
    window_cfg = make_window_cfg()
    d15 = date(2025, 3, 15)
    d16 = date(2025, 3, 16)
    d17 = date(2025, 3, 17)
    d18 = date(2025, 3, 18)
    d19 = date(2025, 3, 19)
    df = pd.concat([
        _synthetic_full_day(date(2025, 3, 13), "H25", window_cfg=window_cfg),
        _synthetic_full_day(date(2025, 3, 14), "H25", window_cfg=window_cfg),
        _synthetic_partial_day(d15, "H25", minutes=[270, 271]),
        _synthetic_partial_day(d16, "M25", minutes=[270, 271]),  # H25 ausente
        _synthetic_full_day(d17, "M25", window_cfg=window_cfg),  # H25 ausente, M25 sesion completa
        _synthetic_full_day(d18, "H25", volume_per_bar=1, window_cfg=window_cfg),
        _synthetic_full_day(d18, "M25", volume_per_bar=2, window_cfg=window_cfg),  # confirma (share 0.667)
        _synthetic_full_day(d19, "M25", window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)

    d17_rows = resolved[resolved["date"] == d17]
    assert d17_rows["contract"].unique().tolist() == ["M25"]
    assert len(d17_rows) == 691

    assert len(events) == 1
    row = events.iloc[0]
    assert row["signal_date"] == d18
    assert row["effective_date"] == d19
    assert resolved[resolved["date"] == d18]["contract"].unique().tolist() == ["H25"]
    assert resolved[resolved["date"] == d19]["contract"].unique().tolist() == ["M25"]


def test_resolve_rollovers_irreversible_ignores_residual_outgoing_data():
    window_cfg = make_window_cfg()
    d1, d2, d3 = date(2024, 3, 18), date(2024, 3, 19), date(2024, 3, 20)
    df = pd.concat([
        _synthetic_full_day(d1, "H24", volume_per_bar=1, window_cfg=window_cfg),
        _synthetic_full_day(d1, "M24", volume_per_bar=2, window_cfg=window_cfg),  # confirma (share 0.667)
        _synthetic_partial_day(d2, "H24", minutes=[270, 271]),  # residual del saliente, ya no activo
        _synthetic_full_day(d2, "M24", volume_per_bar=1, window_cfg=window_cfg),
        _synthetic_full_day(d3, "M24", volume_per_bar=1, window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)
    assert len(events) == 1
    # d2 y d3 deben quedar en M24 (irreversible); el residual H24 de d2 se descarta.
    assert resolved[resolved["date"] == d2]["contract"].unique().tolist() == ["M24"]
    assert resolved[resolved["date"] == d3]["contract"].unique().tolist() == ["M24"]
    residual = discarded[(discarded["date"] == d2) & (discarded["contract"] == "H24")]
    assert len(residual) == 2
    assert (residual["discard_reason"] == "superseded_contract_residual_data").all()


def test_resolve_rollovers_no_rows_lost_or_duplicated():
    window_cfg = make_window_cfg()
    d1 = date(2024, 3, 18)
    df = pd.concat([
        _synthetic_full_day(d1, "H24", volume_per_bar=1, window_cfg=window_cfg),
        _synthetic_full_day(d1, "M24", volume_per_bar=2, window_cfg=window_cfg),
    ])
    resolved, events, ambiguous, discarded = prep.resolve_rollovers(df, window_cfg, min_incoming_share=0.55)
    assert len(resolved) + len(discarded) == len(df)


# ---------------------------------------------------------------------------
# Zona horaria: evaluacion de hipotesis
# ---------------------------------------------------------------------------

def test_evaluate_timezone_hypotheses_prefers_correct_offset():
    # Serie sintetica: evento de "apertura" (pico de volumen) fijado a las
    # 13:30 UTC todos los dias (equivalente a 09:30 America/New_York en un
    # dia de EDT fijo, sin variacion estacional en esta prueba). La hipotesis
    # UTC debe alinear perfectamente (offset 0); America/Chicago (UTC-6/-5)
    # debe quedar peor alineada.
    dates = pd.date_range("2024-06-03", periods=10, freq="D")  # dias de EDT (junio)
    rows = []
    for d in dates:
        for m in range(0, 1440, 1):
            vol = 100
            if m == 13 * 60 + 30:
                vol = 100000  # pico de "apertura" a las 13:30 UTC
            rows.append((d + pd.Timedelta(minutes=m), vol))
    idx = pd.DatetimeIndex([r[0] for r in rows])
    vol = np.array([r[1] for r in rows])

    table = prep.evaluate_timezone_hypotheses(idx, vol, ["UTC", "America/New_York", "America/Chicago"])
    best = table.iloc[0]
    assert best["timezone_candidate"] == "UTC"


# ---------------------------------------------------------------------------
# DST
# ---------------------------------------------------------------------------

def test_dst_check_detects_no_duplicates_when_clean():
    window_cfg = make_window_cfg()
    d = date(2024, 3, 11)  # lunes posterior al spring-forward 2024
    df = _full_day_df(d, window_cfg)
    result = prep.verify_dst_no_duplicates_or_gaps(df, [d])
    assert result["dates_with_duplicates"] == {}


def test_dst_check_flags_real_duplicates():
    window_cfg = make_window_cfg()
    d = date(2024, 3, 11)
    df = _full_day_df(d, window_cfg)
    dup_row = df.iloc[[0]].copy()
    df = pd.concat([df, dup_row])
    result = prep.verify_dst_no_duplicates_or_gaps(df, [d])
    assert str(d) in result["dates_with_duplicates"]


# ---------------------------------------------------------------------------
# Staleness (mismo patron que S00 v2)
# ---------------------------------------------------------------------------

def test_staleness_fields_match_true_when_identical():
    manifest = {"staleness": {
        "raw_parquet_sha256": "h1", "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "s01_v2", "force_rebuild": False,
    }}
    assert prep.staleness_fields_match(manifest, manifest) is True


def test_staleness_fields_match_false_when_raw_hash_changes():
    old = {"staleness": {
        "raw_parquet_sha256": "h1", "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "s01_v2", "force_rebuild": False,
    }}
    new = {"staleness": {**old["staleness"], "raw_parquet_sha256": "h2"}}
    assert prep.staleness_fields_match(old, new) is False


def test_staleness_fields_match_false_when_no_previous_manifest():
    new = {"staleness": {
        "raw_parquet_sha256": "h1", "module_sha256": "m1",
        "config_sha256_normalized": "c1", "schema_expected": ["x"],
        "pipeline_version": "s01_v2", "force_rebuild": False,
    }}
    assert prep.staleness_fields_match(None, new) is False
