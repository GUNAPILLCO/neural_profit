"""Pruebas unitarias rapidas de S02 v2 -- datos sinteticos en memoria."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import s02_intraday_analysis as s02

REGIME_LABELS = {0: "Early_Premarket", 1: "Premarket", 2: "Opening", 3: "Regular", 4: "Closing"}


# ---------------------------------------------------------------------------
# Helpers de construccion sintetica
# ---------------------------------------------------------------------------

def make_bars(rows, start="2024-01-02 04:30:00", tz="America/New_York", contract="H24", price_start=100.0):
    """rows: lista de dicts opcionales por barra: {minute_of_day, date, regime_id,
    consecutive_segment_id, contract, open/high/low/close/volume}."""
    n = len(rows)
    idx = []
    cur = pd.Timestamp(start, tz=tz)
    for i, r in enumerate(rows):
        if "timestamp" in r:
            idx.append(pd.Timestamp(r["timestamp"], tz=tz))
        else:
            idx.append(cur)
            cur = cur + pd.Timedelta(minutes=1)

    prices = [price_start + i for i in range(n)]
    df = pd.DataFrame({
        "date": [r.get("date", idx[i].date()) for i, r in enumerate(rows)],
        "minute_of_day": [r["minute_of_day"] for r in rows],
        "regime_id": [r.get("regime_id", 3) for r in rows],
        "regime_label": [REGIME_LABELS[r.get("regime_id", 3)] for r in rows],
        "consecutive_segment_id": [r.get("consecutive_segment_id", 0) for r in rows],
        "open": [r.get("open", prices[i]) for i, r in enumerate(rows)],
        "high": [r.get("high", prices[i] + 0.5) for i, r in enumerate(rows)],
        "low": [r.get("low", prices[i] - 0.5) for i, r in enumerate(rows)],
        "close": [r.get("close", prices[i]) for i, r in enumerate(rows)],
        "volume": [r.get("volume", 10) for r in rows],
        "contract": [r.get("contract", contract) for r in rows],
    }, index=pd.DatetimeIndex(idx))
    return df


def make_audit_row(d, day_status="full_coverage", eligibility_category="full_day_eligible", regime_consecutive=None):
    regime_consecutive = regime_consecutive or {rid: True for rid in range(5)}
    row = {"date": d, "day_status": day_status, "eligibility_category": eligibility_category}
    for rid in range(5):
        row[f"regime_{rid}_is_consecutive"] = regime_consecutive.get(rid, False)
    return row


# ---------------------------------------------------------------------------
# Poblaciones
# ---------------------------------------------------------------------------

def test_attach_population_tags_and_filter_full_day_eligible():
    d = date(2024, 1, 2)
    df = make_bars([{"minute_of_day": 270 + i, "date": d, "regime_id": 3} for i in range(5)])
    audit = pd.DataFrame([make_audit_row(d)])

    tagged = s02.attach_population_tags(df, audit)
    assert (tagged["eligibility_category"] == "full_day_eligible").all()
    assert tagged["regime_is_consecutive"].all()

    full_day = s02.filter_population(tagged, s02.POPULATION_FULL_DAY)
    assert len(full_day) == 5


def test_attach_population_tags_raises_on_unknown_date():
    d = date(2024, 1, 2)
    df = make_bars([{"minute_of_day": 270, "date": d, "regime_id": 3}])
    audit = pd.DataFrame([make_audit_row(date(2024, 1, 3))])
    with pytest.raises(s02.IngestionError, match="eligibility_category"):
        s02.attach_population_tags(df, audit)


def test_partial_regime_eligible_keeps_only_consecutive_regimes():
    d = date(2024, 1, 2)
    # Regimen 2 (Opening) incompleto/no consecutivo; regimen 3 (Regular) si.
    rows = (
        [{"minute_of_day": 570 + i, "date": d, "regime_id": 2} for i in range(3)]
        + [{"minute_of_day": 630 + i, "date": d, "regime_id": 3} for i in range(5)]
    )
    df = make_bars(rows)
    audit = pd.DataFrame([
        make_audit_row(d, day_status="partial_undetermined", eligibility_category="partial_regime_eligible",
                        regime_consecutive={2: False, 3: True})
    ])
    tagged = s02.attach_population_tags(df, audit)
    partial = s02.filter_population(tagged, s02.POPULATION_PARTIAL_REGIME)
    assert (partial["regime_id"] == 3).all()
    assert len(partial) == 5


def test_not_model_eligible_excluded_from_quantitative_populations():
    d = date(2024, 1, 2)
    df = make_bars([{"minute_of_day": 270, "date": d, "regime_id": 0}])
    audit = pd.DataFrame([make_audit_row(d, day_status="no_data_gap_documented_s00", eligibility_category="not_model_eligible")])
    tagged = s02.attach_population_tags(df, audit)
    for population in (s02.POPULATION_FULL_DAY, s02.POPULATION_PARTIAL_REGIME, s02.POPULATION_DESCRIPTIVE):
        assert s02.filter_population(tagged, population).empty
    assert len(s02.filter_population(tagged, s02.POPULATION_ALL)) == 1


# ---------------------------------------------------------------------------
# Metricas OHLCV: body_signed_pts vs body_abs_pts
# ---------------------------------------------------------------------------

def test_body_signed_and_abs_pts_bullish_bearish_neutral():
    d = date(2024, 1, 2)
    rows = [
        {"minute_of_day": 270, "date": d, "open": 100.0, "close": 103.0, "high": 104.0, "low": 99.0},
        {"minute_of_day": 271, "date": d, "open": 100.0, "close": 97.0, "high": 101.0, "low": 96.0},
        {"minute_of_day": 272, "date": d, "open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0},
    ]
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    assert out["body_signed_pts"].tolist() == pytest.approx([3.0, -3.0, 0.0])
    assert out["body_abs_pts"].tolist() == pytest.approx([3.0, 3.0, 0.0])
    assert out["candle_direction"].tolist() == [1, -1, 0]


def test_ohlcv_metrics_no_leak_across_segment_boundary():
    d = date(2024, 1, 2)
    rows = [
        {"minute_of_day": 270, "date": d, "consecutive_segment_id": 0, "close": 100.0},
        {"minute_of_day": 271, "date": d, "consecutive_segment_id": 0, "close": 101.0},
        {"minute_of_day": 280, "date": d, "consecutive_segment_id": 1, "close": 200.0},
    ]
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    # primera barra de cada segmento no debe tener close anterior valido
    assert pd.isna(out["prev_close_1m"].iloc[0])
    assert out["prev_close_1m"].iloc[1] == 100.0
    assert pd.isna(out["prev_close_1m"].iloc[2])  # no debe heredar 101.0 del segmento anterior


# ---------------------------------------------------------------------------
# Validez de ventanas: off-by-one, cruces de date/segmento/contrato
# ---------------------------------------------------------------------------

def _sequential_rows(n, date_=date(2024, 1, 2), start_minute=270, regime_id=3, contract="H24", segment_id=0):
    return [
        {"minute_of_day": start_minute + i, "date": date_, "regime_id": regime_id, "contract": contract, "consecutive_segment_id": segment_id}
        for i in range(n)
    ]


def test_window_validity_off_by_one_boundary():
    # 10 barras consecutivas: para h=5, la barra en posicion 4 (0-index) es la
    # primera con ventana historica completa (bar_pos >= h-1 = 4).
    df = make_bars(_sequential_rows(10))
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[5])
    hist_valid = valid["hist_window_valid_5m"].tolist()
    assert hist_valid == [False, False, False, False, True, True, True, True, True, True]

    # ventana futura: la ultima barra con ventana futura completa es la de
    # posicion n-1-h = 10-1-5 = 4.
    future_valid = valid["future_window_valid_5m"].tolist()
    assert future_valid == [True, True, True, True, True, False, False, False, False, False]


def test_window_validity_crosses_date_boundary_invalid():
    rows = _sequential_rows(3, date_=date(2024, 1, 2), start_minute=958, segment_id=0) + \
        _sequential_rows(3, date_=date(2024, 1, 3), start_minute=270, segment_id=1)
    # ultimas 2 barras del dia 1 (958, 959) seguidas de 960; luego nuevo dia.
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[2])
    # la barra en minute_of_day=960 (posicion 2) no puede tener ventana futura
    # de h=2 porque cruzaria al dia siguiente (distinto segmento).
    assert valid["future_window_valid_2m"].iloc[2] == False  # noqa: E712
    assert valid["future_invalid_reason_2m"].iloc[2] == "insufficient_bars"


def test_window_validity_crosses_internal_gap_within_same_date():
    # Mismo date, pero un gap de minutos genera dos consecutive_segment_id
    # distintos dentro del mismo dia (jornada parcial con hueco interno).
    rows = (
        _sequential_rows(4, start_minute=270, segment_id=0)
        + _sequential_rows(4, start_minute=290, segment_id=1)
    )
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[3])
    # la barra de posicion 3 (ultima del primer segmento) no puede tener
    # ventana futura de h=3 porque el segmento siguiente es otro tramo.
    assert valid["future_window_valid_3m"].iloc[3] == False  # noqa: E712
    # la barra de posicion 4 (primera del segundo segmento) no puede tener
    # ventana historica de h=3 porque el segmento anterior es otro tramo.
    assert valid["hist_window_valid_3m"].iloc[4] == False  # noqa: E712


def test_window_validity_crosses_contract_change_is_blocking():
    rows = _sequential_rows(4, start_minute=270, contract="H24", segment_id=0) + \
        _sequential_rows(4, start_minute=274, contract="M24", segment_id=0)
    # mismo segmento (minutos estrictamente consecutivos) pero cambia contrato
    # a mitad de camino -- debe invalidar, nunca pasar desapercibido.
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[3])

    # posicion 3 (ultima barra H24): ventana futura de h=3 cruzaria a M24.
    assert valid["future_window_valid_3m"].iloc[3] == False  # noqa: E712
    assert valid["future_invalid_reason_3m"].iloc[3] == "contract_change"

    # posicion 4 (primera barra M24): ventana historica de h=3 cruzaria a H24.
    assert valid["hist_window_valid_3m"].iloc[4] == False  # noqa: E712
    assert valid["hist_invalid_reason_3m"].iloc[4] == "contract_change"

    # posicion 7 (h=3 barras atras, todas M24): valida.
    assert valid["hist_window_valid_3m"].iloc[7] == True  # noqa: E712


def test_window_validity_defensive_check_on_non_consecutive_minutes():
    # minute_of_day corrupto dentro de lo que se declara como un unico
    # consecutive_segment_id (no deberia ocurrir tras S01, pero se verifica).
    rows = [
        {"minute_of_day": 270, "date": date(2024, 1, 2), "consecutive_segment_id": 0},
        {"minute_of_day": 271, "date": date(2024, 1, 2), "consecutive_segment_id": 0},
        {"minute_of_day": 275, "date": date(2024, 1, 2), "consecutive_segment_id": 0},  # salto corrupto
    ]
    df = make_bars(rows)
    validation = s02.validate_intraday_dataset(df)
    assert validation["ok"] is False
    assert "segments_are_strictly_consecutive_minutes" in validation["failed_critical_checks"]


def test_validate_intraday_dataset_does_not_fail_on_legitimate_gap_across_dates_or_segments():
    # Gaps entre segmentos/fechas son legitimos en v2 (jornadas parciales) y
    # NO deben hacer fallar la validacion critica.
    rows = _sequential_rows(3, start_minute=270, segment_id=0) + _sequential_rows(3, start_minute=290, segment_id=1)
    df = make_bars(rows)
    validation = s02.validate_intraday_dataset(df)
    assert validation["ok"] is True
    assert validation["summary"]["n_intraday_gaps_within_date_informational"] == 1


# ---------------------------------------------------------------------------
# Auditoria de rollover a nivel de ventana
# ---------------------------------------------------------------------------

def test_rollover_window_audit_counts_expected_bars():
    rows = _sequential_rows(4, start_minute=270, contract="H24", segment_id=0) + \
        _sequential_rows(4, start_minute=274, contract="M24", segment_id=0)
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[3])
    audit = s02.build_rollover_window_audit(valid, horizons=[3])

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["contract_before"] == "H24"
    assert row["contract_after"] == "M24"
    assert row["intra_segment_transition"] == True  # noqa: E712
    # Transicion H24->M24 entre posicion 3 (H24) y 4 (M24). La condicion de
    # contrato compara SIEMPRE contra el contrato de la barra ancla t (no solo
    # homogeneidad interna de la ventana):
    # Historica en t: requiere contract_run_id[t-2] == contract_run_id[t].
    # Falla (contract_change) para t en {4,5} -> 2 barras.
    assert row["n_bars_hist_invalidated_by_contract"] == 2
    # Futura en t: requiere contract_run_id[t+3] == contract_run_id[t].
    # Falla para t en {1,2,3} -> 3 barras (asimetria esperada: la ventana
    # futura "alcanza" h posiciones desde t, la historica alcanza h-1).
    assert row["n_bars_future_invalidated_by_contract"] == 3


def test_rollover_window_audit_ignores_inter_segment_transitions():
    # Cambio de contrato que coincide con un cambio de segmento (transicion
    # fuera de sesion, el caso normal): no debe contarse como impacto de
    # ventana porque el propio segmento ya invalida el cruce.
    rows = _sequential_rows(3, start_minute=958, contract="H24", segment_id=0) + \
        _sequential_rows(3, start_minute=270, date_=date(2024, 1, 3), contract="M24", segment_id=1)
    df = make_bars(rows)
    out = s02.build_ohlcv_metrics(df)
    valid = s02.build_window_validity(out, horizons=[2])
    audit = s02.build_rollover_window_audit(valid, horizons=[2])
    row = audit.iloc[0]
    assert row["intra_segment_transition"] == False  # noqa: E712
    assert row["n_bars_hist_invalidated_by_contract"] == 0
    assert row["n_bars_future_invalidated_by_contract"] == 0


# ---------------------------------------------------------------------------
# Dependencia temporal: ACF, Ljung-Box, ARCH-LM
# ---------------------------------------------------------------------------

def _config_for_dependence(min_obs=200):
    return {
        "dependence": {
            "series": ["white_noise", "ar1"],
            "max_lag": 5,
            "ljung_box_lags": [5],
            "arch_lm_lags": 5,
            "min_obs_global": min_obs,
            "min_obs_regime": min_obs,
        }
    }


def _synthetic_dependence_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(size=n)
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.8 * ar1[i - 1] + rng.normal()
    idx = pd.date_range("2024-01-02", periods=n, freq="min", tz="America/New_York")
    df = pd.DataFrame({
        "white_noise": white_noise, "ar1": ar1,
        "regime_id": 3, "regime_label": "Regular",
        "eligibility_category": "full_day_eligible", "regime_is_consecutive": True,
        "consecutive_segment_id": 0,
    }, index=idx)
    return df


def test_ljung_box_detects_dependence_in_ar1_not_in_white_noise():
    df = _synthetic_dependence_frame()
    tests = s02.build_dependence_tests_summary(df, [s02.POPULATION_FULL_DAY], _config_for_dependence())
    lb = tests.loc[(tests["test_name"] == "ljung_box") & (tests["scope"] == "global")]

    p_white = lb.loc[lb["series"] == "white_noise", "p_value"].iloc[0]
    p_ar1 = lb.loc[lb["series"] == "ar1", "p_value"].iloc[0]

    assert p_white > 0.05
    assert p_ar1 < 0.01


def test_acf_summary_skips_insufficient_sample():
    df = _synthetic_dependence_frame(n=50)
    config = _config_for_dependence(min_obs=200)
    acf_table = s02.build_acf_summary(df, [s02.POPULATION_FULL_DAY], config)
    assert acf_table.empty


def test_dependence_tests_summary_flags_insufficient_sample():
    df = _synthetic_dependence_frame(n=50)
    config = _config_for_dependence(min_obs=200)
    tests = s02.build_dependence_tests_summary(df, [s02.POPULATION_FULL_DAY], config)
    assert (~tests["sufficient_sample"]).all()
    assert tests["statistic"].isna().all()


# ---------------------------------------------------------------------------
# ACF/Ljung-Box gap-aware: nunca formar pares entre distintos
# consecutive_segment_id (ver revision focalizada tras la aprobacion inicial)
# ---------------------------------------------------------------------------

def test_gap_aware_acf_excludes_cross_segment_pair():
    # Segmento A: [1,2,3,4]. Segmento B: [100,5,6,7], con 100 como outlier
    # deliberado en la PRIMERA posicion de B para que, si se emparejara
    # naivemente con el ultimo valor de A (4), el lag-1 quedara inflado por
    # un par completamente espurio entre dos segmentos distintos.
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0, 7.0])
    segments = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    result = s02._gap_aware_acf(values, segments, max_lag=1)

    # 6 pares reales dentro de cada segmento: (1,2)(2,3)(3,4) + (100,5)(5,6)(6,7).
    # El par espurio (4,100) NUNCA debe contarse.
    assert result["pairs"][0] == 6

    mean = values.mean()
    centered = values - mean
    c0 = float(np.sum(centered ** 2))
    expected_numerator = (
        centered[0] * centered[1] + centered[1] * centered[2] + centered[2] * centered[3]
        + centered[4] * centered[5] + centered[5] * centered[6] + centered[6] * centered[7]
    )
    assert result["acf"][0] == pytest.approx(expected_numerator / c0)

    # La version naive (concatenar y usar TODOS los n-1 pares consecutivos,
    # incluyendo el espurio (4,100)) debe dar un valor DISTINTO -- confirma
    # que el par espurio realmente afecta el resultado si no se excluye.
    naive_numerator = expected_numerator + centered[3] * centered[4]
    naive_r1 = naive_numerator / c0
    assert result["acf"][0] != pytest.approx(naive_r1)


def test_gap_aware_acf_pairs_count_matches_sum_of_within_segment_pairs():
    # 3 segmentos de tamanos 5, 3 y 10 -> a lag=2, pares esperados:
    # max(0,5-2) + max(0,3-2) + max(0,10-2) = 3 + 1 + 8 = 12.
    rng = np.random.default_rng(0)
    sizes = [5, 3, 10]
    segments = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    values = rng.normal(size=segments.shape[0])

    result = s02._gap_aware_acf(values, segments, max_lag=2)
    assert result["pairs"][1] == 3 + 1 + 8


def test_ljung_box_gap_aware_uses_effective_pairs_not_n_minus_k():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0, 7.0])
    segments = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = s02._gap_aware_acf(values, segments, max_lag=2)
    q_stat, p_value, n_pairs_last_lag = s02._ljung_box_gap_aware(result, lags=2)

    # A lag=2: pares reales = (1,3)(2,4) dentro de A + (100,6)(5,7) dentro de
    # B = 4, nunca n-k=6.
    assert result["pairs"][1] == 4
    assert n_pairs_last_lag == 4
    assert np.isfinite(q_stat)
    assert 0.0 <= p_value <= 1.0


def test_clean_series_with_segment_preserves_alignment_after_dropna():
    df = pd.DataFrame({
        "x": [np.nan, 1.0, 2.0, np.nan, 3.0, 4.0],
        "consecutive_segment_id": [0, 0, 0, 1, 1, 1],
    })
    values, segments = s02._clean_series_with_segment(df, "x")
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert segments.tolist() == [0, 0, 1, 1]


def test_acf_summary_reports_effective_pairs_lower_than_naive_n_minus_k_at_day_boundaries():
    # Dos "dias" sinteticos de 100 barras cada uno, mismo patron que produce
    # build_ohlcv_metrics: primera barra de cada segmento sin retorno valido.
    n_per_day = 100
    rng = np.random.default_rng(0)
    values_day1 = rng.normal(size=n_per_day)
    values_day2 = rng.normal(size=n_per_day)
    df = pd.DataFrame({
        "ret_1m": np.concatenate([[np.nan], values_day1[1:], [np.nan], values_day2[1:]]),
        "regime_id": 3, "regime_label": "Regular",
        "consecutive_segment_id": np.concatenate([np.zeros(n_per_day), np.ones(n_per_day)]),
    })
    # Llamado directo al helper de scope (sin pasar por filter_population,
    # que aqui no aplica) para verificar la columna n_pairs_effective.
    rows: list = []
    s02._acf_rows_for_scope(df, "global", None, None, ["ret_1m"], 3, 10, rows, "synthetic")
    acf_table = pd.DataFrame(rows)

    n_obs = acf_table["n_obs"].iloc[0]
    for lag in (1, 2, 3):
        pairs_lag = acf_table.loc[acf_table["lag"] == lag, "n_pairs_effective"].iloc[0]
        naive_pairs = n_obs - lag
        # El par que cruzaria el limite de segmento (ultima barra del dia 1
        # con la primera barra valida del dia 2) nunca debe contarse.
        assert pairs_lag < naive_pairs


# ---------------------------------------------------------------------------
# Gobernanza: staleness
# ---------------------------------------------------------------------------

def test_staleness_fields_match_detects_force_rebuild():
    new_manifest = {"staleness": {"force_rebuild": True, "intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    old = {"staleness": {"intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    assert s02.staleness_fields_match(old, new_manifest) is False


def test_staleness_fields_match_detects_config_change():
    old = {"staleness": {"intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    new_manifest = {"staleness": {"force_rebuild": False, "intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "DIFFERENT", "pipeline_version": "s02_v2"}}
    assert s02.staleness_fields_match(old, new_manifest) is False


def test_staleness_fields_match_true_when_unchanged():
    old = {"staleness": {"intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    new_manifest = {"staleness": {"force_rebuild": False, "intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    assert s02.staleness_fields_match(old, new_manifest) is True


def test_staleness_fields_match_false_when_no_previous_manifest():
    new_manifest = {"staleness": {"force_rebuild": False, "intraday_parquet_sha256": "a", "trading_day_audit_sha256": "b", "module_sha256": "c", "config_sha256_normalized": "d", "pipeline_version": "s02_v2"}}
    assert s02.staleness_fields_match(None, new_manifest) is False
