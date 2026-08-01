"""S01 v2 — preparacion intradia MNQ a partir de data/01_raw/mnq_raw_v2.parquet.

Reglas de alcance (ver plan de auditoria S01 v2):
- No asume la hipotesis UTC: la evalua programaticamente contra
  America/New_York y America/Chicago antes de convertir.
- No asume timestamp_semantics (inicio/cierre de barra): la deja
  unknown_not_confirmed salvo evidencia real.
- No elimina ningun dia por conteo de barras ni por gaps: cataloga cada
  fecha calendario del rango con motivo explicito en
  trading_day_audit_v2.parquet, incluyendo elegibilidad por regimen.
- No construye targets, features, secuencias ni ventanas de modelos --
  solo deja `consecutive_segment_id`, cobertura, gaps y elegibilidad para
  que S02+ los use.
- Ventana y regimenes son los aprobados (config/intraday_config.yaml);
  esta implementacion no vuelve a compararlos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yaml

from src.data.s00_raw_ingestion import (
    IngestionError,
    atomic_write_parquet,
    get_git_provenance,
    normalized_config_bytes,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

MODULE_PATH = Path(__file__).resolve()

NY_TZ = "America/New_York"

DAY_STATUS_FULL = "full_coverage"
DAY_STATUS_PARTIAL_EARLY_CLOSE = "partial_early_close_cme"
DAY_STATUS_PARTIAL_GAP_S00 = "partial_gap_documented_s00"
DAY_STATUS_PARTIAL_UNDETERMINED = "partial_undetermined"
DAY_STATUS_NO_DATA_WEEKEND = "no_data_weekend"
DAY_STATUS_NO_DATA_HOLIDAY = "no_data_cme_holiday"
DAY_STATUS_NO_DATA_GAP_S00 = "no_data_gap_documented_s00"
DAY_STATUS_NO_DATA_UNDETERMINED = "no_data_undetermined"

ELIGIBILITY_FULL = "full_day_eligible"
ELIGIBILITY_EARLY_CLOSE = "early_close_eligible"
ELIGIBILITY_PARTIAL_REGIME = "partial_regime_eligible"
ELIGIBILITY_DESCRIPTIVE = "descriptive_only"
ELIGIBILITY_NOT_ELIGIBLE = "not_model_eligible"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise IngestionError(f"Config vacia o ilegible: {config_path}")
    return config


def get_module_hash() -> str:
    return sha256_file(MODULE_PATH)


# ---------------------------------------------------------------------------
# 1) Validacion programatica de zona horaria (3 hipotesis)
# ---------------------------------------------------------------------------

def _minute_profile(idx_ny: pd.DatetimeIndex, volume: np.ndarray) -> tuple[pd.Series, pd.Series]:
    """Devuelve (volumen_total_por_minuto, cobertura_por_minuto) indexado 0..1439."""
    minute_of_day = idx_ny.hour * 60 + idx_ny.minute
    dates = idx_ny.date
    vol_by_minute = pd.Series(volume, index=minute_of_day).groupby(level=0).sum()
    vol_by_minute = vol_by_minute.reindex(range(1440), fill_value=0)

    n_dates = len(set(dates))
    cov_by_minute = (
        pd.Series(dates, index=minute_of_day).groupby(level=0).nunique()
        if n_dates
        else pd.Series(dtype=int)
    )
    cov_by_minute = cov_by_minute.reindex(range(1440), fill_value=0) / max(n_dates, 1)
    return vol_by_minute, cov_by_minute


def _opening_alignment(vol_by_minute: pd.Series, target_minute: int = 570) -> int:
    window = vol_by_minute.loc[520:640]
    jump = window.diff().dropna()
    if jump.empty or jump.max() <= 0:
        return 999
    best_minute = int(jump.idxmax())
    return abs(best_minute - target_minute)


def _closing_alignment(vol_by_minute: pd.Series, target_minute: int = 960) -> int:
    window = vol_by_minute.loc[900:1020]
    if window.empty or window.max() <= 0:
        return 999
    best_minute = int(window.idxmax())
    return abs(best_minute - target_minute)


def _maintenance_break_alignment(cov_by_minute: pd.Series, target_minute: int = 1020, threshold: float = 0.05) -> int:
    window = cov_by_minute.loc[990:1140]
    below = window[window < threshold]
    if below.empty:
        return 999
    first_break_minute = int(below.index[0])
    return abs(first_break_minute - target_minute)


def evaluate_timezone_hypotheses(
    raw_index: pd.DatetimeIndex,
    raw_volume: np.ndarray,
    candidates: list[str],
) -> pd.DataFrame:
    """Evalua cada hipotesis de zona horaria de origen contra eventos
    estructurales conocidos del mercado, sin asumir ninguna de antemano."""
    rows = []
    for candidate in candidates:
        if candidate == NY_TZ:
            localized = raw_index.tz_localize(candidate, ambiguous="NaT", nonexistent="NaT")
            n_dropped = int(localized.isna().sum())
            idx_ny_full = localized.tz_convert(NY_TZ)
        else:
            localized = raw_index.tz_localize(candidate, ambiguous="NaT", nonexistent="NaT")
            n_dropped = int(localized.isna().sum())
            idx_ny_full = localized.tz_convert(NY_TZ)

        valid_mask = ~pd.isna(idx_ny_full)
        idx_ny = idx_ny_full[valid_mask]
        vol = raw_volume[valid_mask]

        month = idx_ny.month
        edt_mask = np.isin(month, [4, 5, 6, 7, 8, 9, 10])
        est_mask = np.isin(month, [12, 1, 2])

        vol_all, cov_all = _minute_profile(idx_ny, vol)
        opening_all = _opening_alignment(vol_all)
        closing_all = _closing_alignment(vol_all)
        maint_all = _maintenance_break_alignment(cov_all)

        vol_edt, _ = _minute_profile(idx_ny[edt_mask], vol[edt_mask])
        vol_est, _ = _minute_profile(idx_ny[est_mask], vol[est_mask])
        opening_edt = _opening_alignment(vol_edt)
        opening_est = _opening_alignment(vol_est)
        closing_edt = _closing_alignment(vol_edt)
        closing_est = _closing_alignment(vol_est)

        dst_consistency = abs(opening_edt - opening_est) + abs(closing_edt - closing_est)
        ambiguous_penalty = min(n_dropped, 100) * 0.1

        total_score = opening_all + closing_all + maint_all + dst_consistency + ambiguous_penalty

        if total_score <= 5:
            confidence = "high"
        elif total_score <= 30:
            confidence = "medium"
        else:
            confidence = "low"

        conclusion = (
            f"opening off by {opening_all}min, closing off by {closing_all}min, "
            f"maintenance-break off by {maint_all}min, dst_consistency={dst_consistency}min, "
            f"{n_dropped} rows ambiguous/nonexistent under this hypothesis"
        )

        rows.append({
            "timezone_candidate": candidate,
            "opening_alignment": opening_all,
            "closing_alignment": closing_all,
            "maintenance_break_alignment": maint_all,
            "dst_consistency": dst_consistency,
            "ambiguous_nonexistent_rows": n_dropped,
            "total_alignment_score": round(total_score, 2),
            "conclusion": conclusion,
            "confidence_level": confidence,
        })

    return pd.DataFrame(rows).sort_values("total_alignment_score").reset_index(drop=True)


def select_timezone(tz_table: pd.DataFrame) -> dict[str, Any]:
    best = tz_table.iloc[0]
    return {
        "timezone_selected": best["timezone_candidate"],
        "timezone_validation_status": "empirically_supported" if best["confidence_level"] in ("high", "medium") else "inconclusive",
        "timezone_provider_confirmation": False,
        "timezone_evidence": "inferred_from_market_structure_and_dst",
        "total_alignment_score": float(best["total_alignment_score"]),
        "confidence_level": best["confidence_level"],
    }


def convert_to_ny(raw_index: pd.DatetimeIndex, selected_tz: str) -> pd.DatetimeIndex:
    localized = raw_index.tz_localize(selected_tz)
    return localized.tz_convert(NY_TZ)


# ---------------------------------------------------------------------------
# 2) Ventana y regimenes (sin ruta default)
# ---------------------------------------------------------------------------

def filter_window(df_ny: pd.DataFrame, window_cfg: dict[str, Any]) -> pd.DataFrame:
    minute_of_day = df_ny.index.hour * 60 + df_ny.index.minute
    mask = (minute_of_day >= window_cfg["start_minute"]) & (minute_of_day <= window_cfg["end_minute"])
    out = df_ny.loc[mask].copy()
    out["minute_of_day"] = minute_of_day[mask]
    out["date"] = out.index.date
    return out


def build_regime_lookup(regimes_cfg: list[dict[str, Any]], window_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[int, str]]:
    """Construye una tabla de lookup minuto->regime_id para toda la ventana.
    No hay ruta default: cualquier minuto de la ventana sin regimen asignado
    hace fallar la construccion (bug historico C2)."""
    lookup = np.full(1440, -1, dtype=np.int16)
    labels: dict[int, str] = {}
    for regime in regimes_cfg:
        rid = regime["regime_id"]
        labels[rid] = regime["label"]
        lookup[regime["start_minute"]: regime["end_minute"] + 1] = rid

    window_minutes = np.arange(window_cfg["start_minute"], window_cfg["end_minute"] + 1)
    unassigned = window_minutes[lookup[window_minutes] == -1]
    if len(unassigned) > 0:
        raise IngestionError(
            f"Minutos de la ventana sin regimen asignado (ruta default detectada): {unassigned.tolist()}"
        )
    return lookup, labels


def assign_regime(minute_of_day: np.ndarray, lookup: np.ndarray, labels: dict[int, str]) -> tuple[np.ndarray, np.ndarray]:
    regime_id = lookup[minute_of_day]
    if (regime_id == -1).any():
        bad = np.unique(minute_of_day[regime_id == -1])
        raise IngestionError(f"Filas con minute_of_day fuera de cualquier regimen: {bad.tolist()}")
    regime_label = np.array([labels[r] for r in regime_id])
    return regime_id.astype(np.int8), regime_label


# ---------------------------------------------------------------------------
# 2-bis) Resolucion de rollover: una serie principal, un contrato por fecha
# ---------------------------------------------------------------------------
#
# Reglas (ver encargo de resolucion de rollover):
# 1. Se detectan fechas con dos contratos dentro de la ventana 04:30-16:00.
# 2. El volumen se compara unicamente sobre los minutos compartidos por
#    ambos contratos esa fecha.
# 3. El contrato entrante confirma el cruce cuando alcanza >= min_incoming_share
#    del volumen compartido.
# 4. Solo una sesion compartida COMPLETA (los 691 minutos de la ventana,
#    para ambos contratos) puede confirmar el cruce -- una sesion parcial
#    nunca confirma, aunque el share ya supere el umbral.
# 5. El nuevo contrato se aplica desde la siguiente jornada OBSERVADA (la
#    fecha de la senal misma sigue usando el contrato saliente).
# 6. El cambio es irreversible: una vez confirmado, el contrato activo
#    nunca retrocede a uno de rango anterior, aunque reaparezca solo o con
#    mayor volumen residual.
# 7. Una sola sesion confirmante alcanza; no se exige confirmacion doble.
# 8. Nunca se mezclan contratos dentro de una misma fecha (se elige
#    exactamente uno; el resto se descarta a un artefacto de auditoria).
# 9. No se promedia OHLCV ni se crean barras sinteticas: se seleccionan
#    filas reales de un unico contrato.
# 10. Toda fila descartada por esta resolucion queda trazada.
# 11. Regla de respaldo (fallback): si el contrato activo tiene EXACTAMENTE
#    0 barras una fecha, pero el contrato entrante si tiene barras esa
#    fecha, se selecciona la cobertura real del entrante para ESA fecha
#    puntual unicamente (motivo `active_contract_no_data_fallback_to_incoming`).
#    No mezcla contratos (el activo no aporta nada ese dia), no crea
#    barras sinteticas, y NO adelanta formalmente el contrato activo para
#    fechas siguientes -- el cruce formal sigue dependiendo exclusivamente
#    de la regla 3/4 (sesion compartida completa + umbral de volumen). La
#    jornada se clasifica despues con la logica estandar de
#    build_trading_day_audit segun la cobertura real del entrante (691
#    barras -> full_coverage; cobertura parcial -> clasificacion parcial
#    estandar).
#
# Las fechas de transicion NO se hardcodean: se derivan de los datos. Las
# fechas del encargo (Z24->H25, H25->M25, M26->U26, 2025-03-15, 2026-06-11,
# 2025-03-17 como caso de regresion de la regla 11) se usan solo como
# pruebas de regresion sobre el resultado de este algoritmo.

ROLLOVER_REASON_SINGLE_CONTRACT = "single_contract_no_ambiguity"
ROLLOVER_REASON_CONFIRMED_SIGNAL = "confirmed_rollover_signal"
ROLLOVER_REASON_NOT_FULL_SESSION = "ambiguous_not_full_691_session"
ROLLOVER_REASON_BELOW_THRESHOLD = "ambiguous_full_session_below_threshold"
ROLLOVER_REASON_SUPERSEDED_RESIDUAL = "superseded_contract_residual_data"
ROLLOVER_REASON_NO_DATA_FALLBACK = "active_contract_no_data_fallback_to_incoming"


def _full_window_minute_set(window_cfg: dict[str, Any]) -> frozenset[int]:
    return frozenset(range(window_cfg["start_minute"], window_cfg["end_minute"] + 1))


def resolve_rollovers(
    df_window: pd.DataFrame,
    window_cfg: dict[str, Any],
    min_incoming_share: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye una serie principal con un unico contrato por fecha.

    No selecciona contrato en S00 y no promedia OHLCV: para cada fecha con
    mas de un contrato presente, elige las filas reales de exactamente un
    contrato y traza el resto como descartado.

    Devuelve (resolved_df, events_df, ambiguous_df, discarded_df):
    - resolved_df: mismo esquema que df_window, un contrato por `date`.
    - events_df: una fila por transicion CONFIRMADA (senal, saliente,
      entrante, jornada efectiva, evidencia de volumen).
    - ambiguous_df: una fila por cada fecha con mas de un contrato
      presente, confirme o no, con el detalle completo de la evaluacion.
      Distingue explicitamente tres conceptos que NO son intercambiables:
        * formal_active_contract: el contrato activo segun el estado
          irreversible del algoritmo (regla 6) al momento de evaluar esta
          fecha -- puede seguir siendo el saliente incluso si esta fecha
          usa datos del entrante (ver selection_reason).
        * selected_contract_for_date: el contrato cuyas barras reales
          quedan en resolved_df para ESTA fecha puntual (puede diferir de
          formal_active_contract, p.ej. bajo la regla de respaldo 11).
        * selection_reason: por que se eligio ese contrato para esta
          fecha (confirmed_rollover_signal, active_contract_no_data_fallback_to_incoming,
          superseded_contract_residual_data, etc. -- ver constantes
          ROLLOVER_REASON_*).
    - discarded_df: todas las filas removidas por esta resolucion, con
      `discard_reason` (mismo vocabulario que selection_reason).
    """
    full_minutes = _full_window_minute_set(window_cfg)

    if df_window.empty:
        empty = df_window.copy()
        return (
            empty,
            pd.DataFrame(columns=[
                "signal_date", "outgoing_contract", "incoming_contract",
                "effective_date", "shared_minutes", "outgoing_shared_volume",
                "incoming_shared_volume", "incoming_share",
            ]),
            pd.DataFrame(columns=[
                "date", "formal_active_contract", "candidate_contract",
                "outgoing_observed_minutes", "incoming_observed_minutes",
                "shared_minutes", "both_full_691_session",
                "outgoing_shared_volume", "incoming_shared_volume",
                "incoming_share", "is_confirming_session", "confirmed_here",
                "selected_contract_for_date", "selection_reason",
            ]),
            empty.assign(discard_reason=pd.Series(dtype=str)),
        )

    # Orden cronologico real de los contratos, derivado de los datos (nunca
    # hardcodeado): la primera vez que cada contrato aparece en el indice.
    first_seen = df_window.groupby("contract").apply(lambda g: g.index.min(), include_groups=False)
    contract_order = list(first_seen.sort_values().index)
    contract_rank = {c: i for i, c in enumerate(contract_order)}

    # Ultima fecha con datos de cada contrato en TODA la serie cruda (antes
    # de resolver). Distingue un handoff limpio y permanente (el saliente ya
    # no vuelve a tener barras nunca mas -> avance sin ambiguedad real) de un
    # simple hueco de un dia dentro de una ventana de solapamiento activa
    # (el saliente reaparece mas adelante -> sigue pendiente de confirmacion).
    last_seen_date = df_window.groupby("contract")["date"].max().to_dict()

    by_date = {d: g for d, g in df_window.groupby("date")}
    dates_sorted = sorted(by_date.keys())

    active_contract = contract_order[0]
    events: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    discarded_frames: list[pd.DataFrame] = []
    resolved_frames: list[pd.DataFrame] = []

    for d in dates_sorted:
        day_df = by_date[d]
        present = sorted(day_df["contract"].unique().tolist(), key=lambda c: contract_rank[c])

        if len(present) == 1:
            only = present[0]
            if only == active_contract:
                resolved_frames.append(day_df)
            elif contract_rank[only] < contract_rank[active_contract]:
                # Datos residuales de un contrato YA SUPERADO (regla 6,
                # irreversibilidad): el activo no tiene barras esta fecha,
                # asi que la serie resuelta queda sin datos ese dia y el
                # residual se traza como descartado, nunca se revierte.
                discarded_today = day_df.copy()
                discarded_today["discard_reason"] = ROLLOVER_REASON_SUPERSEDED_RESIDUAL
                discarded_frames.append(discarded_today)
            elif d > last_seen_date[active_contract]:
                # Handoff limpio y permanente: el contrato activo ya no
                # vuelve a tener NINGUNA barra en el resto de la serie (no
                # hubo, o ya termino, una ventana de solapamiento real que
                # confirmar). No hay eleccion ambigua que hacer -- el activo
                # avanza directamente al unico contrato disponible.
                resolved_frames.append(day_df)
                active_contract = only
            else:
                # Regla de respaldo (regla 11): el activo tiene EXACTAMENTE
                # 0 barras esta fecha (hueco dentro de una ventana de
                # solapamiento todavia sin confirmar), pero el entrante SI
                # tiene barras. Se selecciona la cobertura real del
                # entrante para esta fecha puntual -- no se mezcla con el
                # activo (que no aporta nada ese dia), no se crean barras
                # sinteticas, y el contrato activo NO avanza: el cruce
                # formal para las fechas siguientes sigue dependiendo
                # exclusivamente de la confirmacion por volumen (reglas 3-4).
                resolved_frames.append(day_df)
                ambiguous_rows.append({
                    "date": d,
                    "formal_active_contract": active_contract,
                    "candidate_contract": only,
                    "outgoing_observed_minutes": 0,
                    "incoming_observed_minutes": len(set(day_df["minute_of_day"].tolist())),
                    "shared_minutes": 0,
                    "both_full_691_session": False,
                    "outgoing_shared_volume": 0.0,
                    "incoming_shared_volume": 0.0,
                    "incoming_share": None,
                    "is_confirming_session": False,
                    "confirmed_here": False,
                    "selected_contract_for_date": only,
                    "selection_reason": ROLLOVER_REASON_NO_DATA_FALLBACK,
                })
                # active_contract se mantiene sin cambios (regla 11): esto
                # no es una confirmacion de rollover, solo resuelve la
                # fecha puntual sin datos del activo.
            continue

        if len(present) > 2:
            raise IngestionError(
                f"{d}: {len(present)} contratos presentes simultaneamente "
                f"({present}) -- la resolucion de rollover solo contempla "
                "pares consecutivos; requiere revision manual."
            )

        # len(present) == 2: fecha ambigua.
        if active_contract not in present:
            raise IngestionError(
                f"{d}: ninguno de los contratos presentes ({present}) es el "
                f"contrato activo ({active_contract}) -- secuencia inesperada, "
                "requiere revision manual."
            )

        candidate = next(c for c in present if c != active_contract)
        out_g = day_df[day_df["contract"] == active_contract]
        in_g = day_df[day_df["contract"] == candidate]
        out_minutes = set(out_g["minute_of_day"].tolist())
        in_minutes = set(in_g["minute_of_day"].tolist())
        shared_minutes = out_minutes & in_minutes
        both_full = (out_minutes == full_minutes) and (in_minutes == full_minutes)

        out_vol = float(out_g.loc[out_g["minute_of_day"].isin(shared_minutes), "volume"].sum())
        in_vol = float(in_g.loc[in_g["minute_of_day"].isin(shared_minutes), "volume"].sum())
        total_shared_vol = out_vol + in_vol
        in_share = (in_vol / total_shared_vol) if total_shared_vol > 0 else None

        # Regla 6 (irreversibilidad): un contrato de rango anterior al
        # activo nunca puede ser evaluado como candidato de cruce, aunque
        # tenga datos residuales ese dia -- solo se descarta.
        candidate_is_forward = contract_rank[candidate] > contract_rank[active_contract]

        is_confirming_session = candidate_is_forward and both_full  # regla 4
        confirmed_here = bool(
            is_confirming_session
            and in_share is not None
            and in_share >= min_incoming_share
        )

        if not candidate_is_forward:
            reason = ROLLOVER_REASON_SUPERSEDED_RESIDUAL
        elif confirmed_here:
            reason = ROLLOVER_REASON_CONFIRMED_SIGNAL
        elif not is_confirming_session:
            reason = ROLLOVER_REASON_NOT_FULL_SESSION
        else:
            reason = ROLLOVER_REASON_BELOW_THRESHOLD

        # Regla 5: la jornada de la SENAL misma sigue usando el contrato
        # saliente. El contrato nuevo solo se activa desde la siguiente
        # jornada observada (se aplica mas abajo, para la iteracion siguiente).
        chosen = active_contract

        ambiguous_rows.append({
            "date": d,
            "formal_active_contract": active_contract,
            "candidate_contract": candidate,
            "outgoing_observed_minutes": len(out_minutes),
            "incoming_observed_minutes": len(in_minutes),
            "shared_minutes": len(shared_minutes),
            "both_full_691_session": both_full,
            "outgoing_shared_volume": out_vol,
            "incoming_shared_volume": in_vol,
            "incoming_share": in_share,
            "is_confirming_session": is_confirming_session,
            "confirmed_here": confirmed_here,
            "selected_contract_for_date": chosen,
            "selection_reason": reason,
        })

        chosen_df = day_df[day_df["contract"] == chosen]
        discarded_today = day_df[day_df["contract"] != chosen].copy()
        discarded_today["discard_reason"] = reason
        discarded_frames.append(discarded_today)
        resolved_frames.append(chosen_df)

        if confirmed_here:
            future_dates = [x for x in dates_sorted if x > d]
            effective_date = future_dates[0] if future_dates else None
            events.append({
                "signal_date": d,
                "outgoing_contract": active_contract,
                "incoming_contract": candidate,
                "effective_date": effective_date,
                "shared_minutes": len(shared_minutes),
                "outgoing_shared_volume": out_vol,
                "incoming_shared_volume": in_vol,
                "incoming_share": in_share,
            })
            active_contract = candidate  # regla 6: irreversible desde aqui

    resolved_df = pd.concat(resolved_frames).sort_index() if resolved_frames else df_window.iloc[0:0].copy()
    events_df = pd.DataFrame.from_records(events)
    ambiguous_df = pd.DataFrame.from_records(ambiguous_rows)
    if discarded_frames:
        discarded_df = pd.concat(discarded_frames).sort_index()
    else:
        discarded_df = df_window.iloc[0:0].copy()
        discarded_df["discard_reason"] = pd.Series(dtype=str)

    # Validacion bloqueante de conservacion (regla 9/10: no se promedia ni
    # se inventan filas, y toda fila removida queda trazada). Si esto
    # falla, alguna fila del insumo se perdio o se duplico silenciosamente
    # entre resolved_df y discarded_df -- debe detener el pipeline, nunca
    # continuar con un dataset inconsistente.
    n_in = len(df_window)
    n_out = len(resolved_df) + len(discarded_df)
    if n_out != n_in:
        raise IngestionError(
            "Violacion de conservacion en resolve_rollovers: "
            f"len(df_window)={n_in} != len(resolved_df)+len(discarded_df)={n_out} "
            f"(resolved={len(resolved_df)}, discarded={len(discarded_df)})."
        )

    return resolved_df, events_df, ambiguous_df, discarded_df


# ---------------------------------------------------------------------------
# 3) Segmentos consecutivos (a nivel de barra, dentro de cada dia)
# ---------------------------------------------------------------------------

def assign_consecutive_segments(df_window: pd.DataFrame) -> np.ndarray:
    """Un id de segmento por corrida ininterrumpida de minutos consecutivos
    dentro de cada `date`. Se reinicia en cada dia y en cada gap > 1 minuto."""
    minute = df_window["minute_of_day"].to_numpy()
    date_arr = df_window["date"].to_numpy()

    order = np.lexsort((minute, date_arr))
    seg_id = np.empty(len(df_window), dtype=np.int32)

    current_seg = 0
    seg_id[order[0]] = current_seg
    for i in range(1, len(order)):
        prev_idx, cur_idx = order[i - 1], order[i]
        same_day = date_arr[cur_idx] == date_arr[prev_idx]
        contiguous = same_day and (minute[cur_idx] - minute[prev_idx] == 1)
        if not contiguous:
            current_seg += 1
        seg_id[cur_idx] = current_seg
    return seg_id


def build_segment_summary(df_window: pd.DataFrame) -> pd.DataFrame:
    """Un resumen por segmento consecutivo: fecha, contrato, minuto de
    inicio/fin y longitud. Persistido para que S02+ nunca tenga que
    recalcular limites de segmento ni cruzarlos por accidente."""
    if df_window.empty:
        return pd.DataFrame(columns=[
            "date", "segment_id", "contract", "start_minute", "end_minute", "length",
        ])

    rows = []
    for (d, seg_id), sub in df_window.groupby(["date", "consecutive_segment_id"]):
        minutes = sub["minute_of_day"].to_numpy()
        contracts = sub["contract"].unique()
        rows.append({
            "date": d,
            "segment_id": int(seg_id),
            "contract": contracts[0] if len(contracts) == 1 else "MIXED",
            "start_minute": int(minutes.min()),
            "end_minute": int(minutes.max()),
            "length": int(len(sub)),
        })
    return pd.DataFrame.from_records(rows).sort_values(["date", "segment_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) Calendario CME_Equity + auditoria de jornadas
# ---------------------------------------------------------------------------

def get_cme_trading_days(start: date_cls, end: date_cls) -> set[date_cls]:
    cal = mcal.get_calendar("CME_Equity")
    sched = cal.schedule(start_date=start, end_date=end)
    return set(sched.index.date)


def get_cme_early_close_dates(start: date_cls, end: date_cls, full_close_time: str = "17:00") -> dict[date_cls, str]:
    cal = mcal.get_calendar("CME_Equity")
    sched = cal.schedule(start_date=start, end_date=end, tz=NY_TZ)
    early = {}
    for ts, row in sched.iterrows():
        close_str = row["market_close"].strftime("%H:%M")
        if close_str != full_close_time:
            early[ts.date()] = close_str
    return early


def _gap_reference_for_date(d: date_cls, known_gaps: list[dict[str, Any]]) -> str | None:
    for gap in known_gaps:
        if gap.get("gap_type") == "recurrent_minor_pattern":
            continue
        start = pd.Timestamp(gap["start"]).date()
        end = pd.Timestamp(gap["end"]).date()
        if start <= d <= end:
            return gap["gap_id"]
    return None


def build_trading_day_audit(
    df_window: pd.DataFrame,
    regimes_cfg: list[dict[str, Any]],
    window_cfg: dict[str, Any],
    known_gaps: list[dict[str, Any]],
    date_range: pd.DatetimeIndex,
    early_close_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    expected_bars = window_cfg["expected_minutes"]
    start_d, end_d = date_range.min().date(), date_range.max().date()
    trading_days = get_cme_trading_days(start_d, end_d)
    early_close_days = get_cme_early_close_dates(start_d, end_d)

    early_close_cfg = early_close_cfg or {"end_minute": 780, "expected_minutes": 511}
    early_close_minutes = frozenset(range(window_cfg["start_minute"], early_close_cfg["end_minute"] + 1))
    early_close_expected_bars = early_close_cfg["expected_minutes"]

    by_date = {d: g for d, g in df_window.groupby("date")}

    records = []
    for ts in date_range:
        d = ts.date()
        sub = by_date.get(d)

        if sub is None or sub.empty:
            observed_bars = 0
            first_time = None
            last_time = None
            consecutive_segment_count = 0
            longest_run = 0
            internal_gap_count = 0
            max_gap_minutes = 0
            missing_minutes = list(range(window_cfg["start_minute"], window_cfg["end_minute"] + 1))
            regime_stats = {r["regime_id"]: {"observed": 0, "consecutive": False} for r in regimes_cfg}
            n_contracts_observed = 0
            is_verified_early_close = False
        else:
            sub = sub.sort_values("minute_of_day")
            observed_bars = len(sub)
            first_time = sub.index.min().strftime("%H:%M")
            last_time = sub.index.max().strftime("%H:%M")
            present_minutes = sub["minute_of_day"].to_numpy()
            all_minutes = set(range(window_cfg["start_minute"], window_cfg["end_minute"] + 1))
            missing_minutes = sorted(all_minutes - set(present_minutes.tolist()))

            n_contracts_observed = int(sub["contract"].nunique())
            if n_contracts_observed > 1:
                raise IngestionError(
                    f"{d}: mas de un contrato presente tras la resolucion de "
                    f"rollover ({sorted(sub['contract'].unique().tolist())}) -- "
                    "invariante violada (regla 8, no mezclar contratos por fecha)."
                )

            seg_sizes = sub.groupby("consecutive_segment_id").size()
            consecutive_segment_count = int(len(seg_sizes))
            longest_run = int(seg_sizes.max())

            diffs = np.diff(present_minutes)
            gaps = diffs[diffs > 1]
            internal_gap_count = int(len(gaps))
            max_gap_minutes = int(gaps.max() - 1) if len(gaps) else 0

            is_verified_early_close = (
                observed_bars == early_close_expected_bars
                and set(present_minutes.tolist()) == early_close_minutes
                and internal_gap_count == 0
            )

            regime_stats = {}
            for r in regimes_cfg:
                rid = r["regime_id"]
                r_expected = r["end_minute"] - r["start_minute"] + 1
                r_sub = sub[(sub["minute_of_day"] >= r["start_minute"]) & (sub["minute_of_day"] <= r["end_minute"])]
                r_observed = len(r_sub)
                r_consecutive = False
                if r_observed == r_expected:
                    r_present = np.sort(r_sub["minute_of_day"].to_numpy())
                    r_consecutive = bool(np.all(np.diff(r_present) == 1)) if r_observed > 1 else True
                regime_stats[rid] = {"observed": r_observed, "expected": r_expected, "consecutive": r_consecutive}

        is_fully_consecutive = (
            observed_bars == expected_bars
            and internal_gap_count == 0
            and n_contracts_observed <= 1
        )
        missing_bars = expected_bars - observed_bars
        coverage_ratio = observed_bars / expected_bars if expected_bars else 0.0

        gap_ref = _gap_reference_for_date(d, known_gaps)

        if d in trading_days:
            calendar_status = "cme_trading_day"
        elif ts.dayofweek >= 5:
            calendar_status = "weekend"
        else:
            calendar_status = "cme_holiday"

        if gap_ref is not None:
            day_status = DAY_STATUS_NO_DATA_GAP_S00 if observed_bars == 0 else DAY_STATUS_PARTIAL_GAP_S00
            inclusion_reason = f"falls within documented S00 gap window ({gap_ref})"
        elif observed_bars == 0:
            if calendar_status == "weekend":
                day_status = DAY_STATUS_NO_DATA_WEEKEND
                inclusion_reason = "weekend, no trading expected"
            elif calendar_status == "cme_holiday":
                day_status = DAY_STATUS_NO_DATA_HOLIDAY
                inclusion_reason = "CME_Equity holiday, no trading expected"
            else:
                day_status = DAY_STATUS_NO_DATA_UNDETERMINED
                inclusion_reason = "CME_Equity marks this as a trading day but zero bars observed; cause not determined by S01"
        elif is_fully_consecutive:
            day_status = DAY_STATUS_FULL
            inclusion_reason = f"{expected_bars}/{expected_bars} bars, fully consecutive, single contract"
        elif d in early_close_days and is_verified_early_close:
            day_status = DAY_STATUS_PARTIAL_EARLY_CLOSE
            inclusion_reason = (
                f"CME_Equity early close scheduled at {early_close_days[d]} ET; "
                f"verified {early_close_expected_bars} consecutive bars "
                f"{window_cfg['start_time']}-{early_close_cfg.get('end_time', '13:00:00')}"
            )
        else:
            day_status = DAY_STATUS_PARTIAL_UNDETERMINED
            inclusion_reason = f"{observed_bars}/{expected_bars} bars observed; no known reason for the shortfall"

        if gap_ref is not None:
            eligibility_category = ELIGIBILITY_NOT_ELIGIBLE
            is_model_eligible = False
            eligibility_reason = f"gap_documented_s00:{gap_ref}"
        elif observed_bars == 0:
            eligibility_category = ELIGIBILITY_NOT_ELIGIBLE
            is_model_eligible = False
            eligibility_reason = f"no_data:{day_status}"
        elif is_fully_consecutive:
            eligibility_category = ELIGIBILITY_FULL
            is_model_eligible = True
            eligibility_reason = "full window coverage, fully consecutive, single contract"
        elif day_status == DAY_STATUS_PARTIAL_EARLY_CLOSE:
            eligibility_category = ELIGIBILITY_EARLY_CLOSE
            is_model_eligible = True
            eligibility_reason = "verified CME early close session, not included in the default full-window population"
        elif any(rs.get("consecutive") for rs in regime_stats.values()):
            eligible_regimes = [rid for rid, rs in regime_stats.items() if rs.get("consecutive")]
            eligibility_category = ELIGIBILITY_PARTIAL_REGIME
            is_model_eligible = True
            eligibility_reason = f"regime(s) fully covered and consecutive: {eligible_regimes}"
        else:
            eligibility_category = ELIGIBILITY_DESCRIPTIVE
            is_model_eligible = False
            eligibility_reason = "partial data present, no regime fully consecutive"

        record = {
            "date": d,
            "calendar_status": calendar_status,
            "observed_bars": observed_bars,
            "expected_bars": expected_bars,
            "missing_bars": missing_bars,
            "coverage_ratio": coverage_ratio,
            "first_observed_time": first_time,
            "last_observed_time": last_time,
            "n_contracts_observed": n_contracts_observed,
            "is_fully_consecutive": is_fully_consecutive,
            "consecutive_segment_count": consecutive_segment_count,
            "longest_consecutive_run": longest_run,
            "internal_gap_count": internal_gap_count,
            "maximum_gap_minutes": max_gap_minutes,
            # Serializado como JSON (no lista nativa): pyarrow reconstruye
            # columnas list<int> como numpy.ndarray al releer, lo que rompe
            # la comparacion de igualdad logica en la escritura atomica
            # (ver atomic_write_parquet en src/data/s00_raw_ingestion.py,
            # que no se puede modificar). Un string JSON round-tripea exacto.
            "missing_minutes": json.dumps(missing_minutes),
            "day_status": day_status,
            "inclusion_reason": inclusion_reason,
            "s00_gap_reference": gap_ref,
            "is_model_eligible": is_model_eligible,
            "eligibility_category": eligibility_category,
            "eligibility_reason": eligibility_reason,
        }
        for r in regimes_cfg:
            rid = r["regime_id"]
            rs = regime_stats[rid]
            r_expected = r["end_minute"] - r["start_minute"] + 1
            r_observed = rs.get("observed", 0)
            record[f"regime_{rid}_observed_bars"] = r_observed
            record[f"regime_{rid}_expected_bars"] = r_expected
            record[f"regime_{rid}_missing_bars"] = r_expected - r_observed
            record[f"regime_{rid}_coverage_ratio"] = r_observed / r_expected if r_expected else 0.0
            record[f"regime_{rid}_is_consecutive"] = bool(rs.get("consecutive", False))

        records.append(record)

    return pd.DataFrame.from_records(records)


def build_regime_distribution(df_window: pd.DataFrame) -> pd.DataFrame:
    agg = df_window.groupby(["regime_id", "regime_label"]).agg(
        n_bars=("close", "size"),
        n_days=("date", "nunique"),
    ).reset_index()
    return agg.sort_values("regime_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) DST: verificacion explicita
# ---------------------------------------------------------------------------

def get_dst_transition_dates(start_year: int, end_year: int) -> list[tuple[str, date_cls]]:
    transitions = []
    for year in range(start_year, end_year + 1):
        march_sundays = pd.date_range(f"{year}-03-01", f"{year}-03-31", freq="W-SUN")
        transitions.append(("spring_forward", march_sundays[1].date()))
        nov_sundays = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="W-SUN")
        transitions.append(("fall_back", nov_sundays[0].date()))
    return transitions


def verify_dst_no_duplicates_or_gaps(df_window: pd.DataFrame, dates: list[date_cls]) -> dict[str, Any]:
    """Para cada fecha, confirma que minute_of_day no tiene duplicados dentro
    de la ventana (seria la firma de una conversion DST mal manejada)."""
    issues = {}
    by_date = {d: g for d, g in df_window.groupby("date")}
    for d in dates:
        sub = by_date.get(d)
        if sub is None or sub.empty:
            continue
        dup = sub["minute_of_day"].duplicated().sum()
        if dup > 0:
            issues[str(d)] = int(dup)
    return {"dates_checked": len(dates), "dates_with_duplicates": issues}


# ---------------------------------------------------------------------------
# 6) Manifest, summary, staleness (mismo patron que S00 v2)
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    config: dict[str, Any],
    raw_manifest: dict[str, Any],
    raw_parquet_path: Path,
    tz_table: pd.DataFrame,
    tz_selection: dict[str, Any],
    df_window: pd.DataFrame,
    audit_df: pd.DataFrame,
    regime_dist: pd.DataFrame,
    dst_check: dict[str, Any],
    rollover_events: pd.DataFrame,
    rollover_ambiguous: pd.DataFrame,
    rollover_discarded: pd.DataFrame,
    n_rows_pre_rollover: int,
    repo_root: Path,
) -> dict[str, Any]:
    provenance = get_git_provenance(repo_root)

    day_status_counts = audit_df["day_status"].value_counts().to_dict()
    eligibility_counts = audit_df["eligibility_category"].value_counts().to_dict()

    n_rows_resolved = int(len(df_window))
    n_rows_discarded = int(len(rollover_discarded))
    conservation_check_passed = bool(n_rows_pre_rollover == n_rows_resolved + n_rows_discarded)
    if not conservation_check_passed:
        raise IngestionError(
            "Violacion de conservacion detectada al construir el manifest: "
            f"n_rows_pre_rollover={n_rows_pre_rollover} != "
            f"n_rows_resolved({n_rows_resolved}) + n_rows_discarded({n_rows_discarded})."
        )

    rollover_summary = {
        "n_ambiguous_dates": int(len(rollover_ambiguous)),
        "n_confirmed_transitions": int(len(rollover_events)),
        "n_rows_pre_rollover": n_rows_pre_rollover,
        "n_rows_resolved": n_rows_resolved,
        "n_discarded_rows": n_rows_discarded,
        "conservation_check": (
            f"{n_rows_pre_rollover} == {n_rows_resolved} + {n_rows_discarded}"
        ),
        "conservation_check_passed": conservation_check_passed,
        "events": json.loads(
            rollover_events.assign(
                signal_date=lambda d: d["signal_date"].astype(str),
                effective_date=lambda d: d["effective_date"].astype(str),
            ).to_json(orient="records")
        ) if not rollover_events.empty else [],
    }

    manifest = {
        "pipeline_version": config["pipeline_version"],
        "staleness": {
            "raw_parquet_sha256": sha256_file(raw_parquet_path),
            "raw_manifest_module_sha256": raw_manifest.get("staleness", {}).get("module_sha256"),
            "module_sha256": get_module_hash(),
            "config_sha256_normalized": sha256_bytes(normalized_config_bytes(config)),
            "schema_expected": list(df_window.columns),
            "pipeline_version": config["pipeline_version"],
            "force_rebuild": bool(config.get("force_rebuild", False)),
        },
        "provenance_metadata_only": {
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
        },
        "window": config["window"],
        "regimes": config["regimes"],
        "timezone": tz_selection,
        "timezone_validation_table": tz_table.to_dict(orient="records"),
        "timestamp_semantics": config["timestamp_semantics"],
        "calendar": config["calendar"],
        "known_gaps": config["known_gaps"],
        "dst_check": dst_check,
        "dataset": {
            "n_rows": int(len(df_window)),
            "n_columns": int(df_window.shape[1]),
            "columns": list(df_window.columns),
            "first_timestamp": str(df_window.index.min()),
            "last_timestamp": str(df_window.index.max()),
            "n_dates": int(audit_df.shape[0]),
        },
        "day_classification": {
            "by_day_status": day_status_counts,
            "by_eligibility_category": eligibility_counts,
        },
        "regime_distribution": regime_dist.to_dict(orient="records"),
        "rollover": rollover_summary,
    }
    return manifest


def staleness_fields_match(old: dict[str, Any] | None, new_manifest: dict[str, Any]) -> bool:
    if old is None:
        return False
    old_st = old.get("staleness", {})
    new_st = new_manifest["staleness"]
    if new_st["force_rebuild"]:
        return False
    keys = [
        "raw_parquet_sha256", "module_sha256", "config_sha256_normalized",
        "schema_expected", "pipeline_version",
    ]
    return all(old_st.get(k) == new_st.get(k) for k in keys)


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "mnq_intraday_v2",
        "pipeline_version": manifest["pipeline_version"],
        "shape": [manifest["dataset"]["n_rows"], manifest["dataset"]["n_columns"]],
        "columns": manifest["dataset"]["columns"],
        "window": manifest["window"],
        "timezone_selected": manifest["timezone"]["timezone_selected"],
        "timezone_validation_status": manifest["timezone"]["timezone_validation_status"],
        "timezone_provider_confirmation": manifest["timezone"]["timezone_provider_confirmation"],
        "timezone_evidence": manifest["timezone"]["timezone_evidence"],
        "timestamp_semantics": manifest["timestamp_semantics"],
        "datetime_min": manifest["dataset"]["first_timestamp"],
        "datetime_max": manifest["dataset"]["last_timestamp"],
        "n_dates": manifest["dataset"]["n_dates"],
        "day_classification": manifest["day_classification"],
        "regime_distribution": manifest["regime_distribution"],
        "dst_check": manifest["dst_check"],
        "rollover": manifest["rollover"],
        "note": (
            "La ventana 04:30-16:00 y sus limites no son quiebres estructurales "
            "optimos (ver reports/stage_reports/S01_v2_report.md); se conservan "
            "por continuidad metodologica. timezone_selected es evidencia "
            "empirica, no confirmacion documental del proveedor."
        ),
    }


# ---------------------------------------------------------------------------
# 7) Orquestacion
# ---------------------------------------------------------------------------

@dataclass
class S01Result:
    reused_existing: bool
    df_window: pd.DataFrame
    audit_df: pd.DataFrame
    regime_dist: pd.DataFrame
    tz_table: pd.DataFrame
    manifest: dict[str, Any]
    summary: dict[str, Any]
    rollover_events: pd.DataFrame
    rollover_ambiguous: pd.DataFrame
    rollover_discarded: pd.DataFrame
    segment_summary: pd.DataFrame
    parquet_path: Path
    parquet_sha256: str
    audit_path: Path
    regime_dist_path: Path
    manifest_path: Path
    summary_path: Path
    tz_validation_path: Path
    rollover_events_path: Path
    rollover_ambiguous_path: Path
    rollover_discarded_path: Path
    segment_summary_path: Path


def run_s01_preparation(
    project_root: Path,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    force_rebuild: bool | None = None,
) -> S01Result:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path) if config_path else project_root / "config" / "intraday_config.yaml"
    config = load_config(config_path)
    if force_rebuild is not None:
        config["force_rebuild"] = force_rebuild

    raw_parquet_path = project_root / config["input"]["raw_parquet"]
    raw_manifest_path = project_root / config["input"]["raw_manifest"]
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))

    artifacts_dir = Path(output_dir) if output_dir else project_root / config["artifacts"]["intraday_dir"]
    parquet_path = artifacts_dir / config["artifacts"]["parquet_name"]
    summary_path = artifacts_dir / config["artifacts"]["summary_name"]
    manifest_path = artifacts_dir / config["artifacts"]["manifest_name"]
    audit_path = artifacts_dir / config["artifacts"]["trading_day_audit_name"]
    regime_dist_path = artifacts_dir / config["artifacts"]["regime_distribution_name"]
    tz_validation_path = artifacts_dir / config["artifacts"]["tz_validation_name"]
    rollover_events_path = artifacts_dir / config["artifacts"]["rollover_events_name"]
    rollover_ambiguous_path = artifacts_dir / config["artifacts"]["rollover_ambiguous_dates_name"]
    rollover_discarded_path = artifacts_dir / config["artifacts"]["rollover_discarded_rows_name"]
    segment_summary_path = artifacts_dir / config["artifacts"]["consecutive_segments_name"]

    df_raw = pd.read_parquet(raw_parquet_path)

    tz_table = evaluate_timezone_hypotheses(
        df_raw.index, df_raw["volume"].to_numpy(), config["timezone"]["candidates"]
    )
    tz_selection = select_timezone(tz_table)

    idx_ny = convert_to_ny(df_raw.index, tz_selection["timezone_selected"])
    df_ny = df_raw.copy()
    df_ny.index = idx_ny

    df_window_raw = filter_window(df_ny, config["window"])

    # Resolucion de rollover: construye una serie principal con un unico
    # contrato por fecha ANTES de asignar regimenes, segmentos o auditar
    # jornadas -- todo lo que sigue ya opera sobre una serie sin ambiguedad
    # de contrato (ver reglas 1-10 en resolve_rollovers).
    rollover_cfg = config.get("rollover", {})
    df_window, rollover_events, rollover_ambiguous, rollover_discarded = resolve_rollovers(
        df_window_raw, config["window"],
        min_incoming_share=rollover_cfg.get("min_incoming_share", 0.55),
    )

    lookup, labels = build_regime_lookup(config["regimes"], config["window"])
    regime_id, regime_label = assign_regime(df_window["minute_of_day"].to_numpy(), lookup, labels)
    df_window["regime_id"] = regime_id
    df_window["regime_label"] = regime_label

    df_window["consecutive_segment_id"] = assign_consecutive_segments(df_window)
    segment_summary = build_segment_summary(df_window)

    date_range = pd.date_range(df_ny.index.min().date(), df_ny.index.max().date(), freq="D")
    audit_df = build_trading_day_audit(
        df_window, config["regimes"], config["window"], config["known_gaps"], date_range,
        early_close_cfg=config.get("early_close"),
    )
    regime_dist = build_regime_distribution(df_window)

    dst_dates = [d for _, d in get_dst_transition_dates(2020, 2026)]
    dst_check = verify_dst_no_duplicates_or_gaps(df_window, dst_dates)

    manifest = build_manifest(
        config=config, raw_manifest=raw_manifest, raw_parquet_path=raw_parquet_path,
        tz_table=tz_table, tz_selection=tz_selection, df_window=df_window,
        audit_df=audit_df, regime_dist=regime_dist, dst_check=dst_check,
        rollover_events=rollover_events, rollover_ambiguous=rollover_ambiguous,
        rollover_discarded=rollover_discarded, n_rows_pre_rollover=int(len(df_window_raw)),
        repo_root=project_root,
    )

    existing_manifest = None
    if manifest_path.exists() and parquet_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_manifest = None

    final_df = df_window[[
        "date", "minute_of_day", "regime_id", "regime_label", "consecutive_segment_id",
        "open", "high", "low", "close", "volume", "contract",
    ]].copy()

    if staleness_fields_match(existing_manifest, manifest):
        reused_df = pd.read_parquet(parquet_path)
        reused_audit = pd.read_parquet(audit_path) if audit_path.exists() else audit_df
        reused_regime_dist = pd.read_parquet(regime_dist_path) if regime_dist_path.exists() else regime_dist
        reused_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else build_summary(existing_manifest)
        reused_events = pd.read_parquet(rollover_events_path) if rollover_events_path.exists() else rollover_events
        reused_ambiguous = pd.read_parquet(rollover_ambiguous_path) if rollover_ambiguous_path.exists() else rollover_ambiguous
        reused_discarded = pd.read_parquet(rollover_discarded_path) if rollover_discarded_path.exists() else rollover_discarded
        reused_segments = pd.read_parquet(segment_summary_path) if segment_summary_path.exists() else segment_summary
        return S01Result(
            reused_existing=True, df_window=reused_df, audit_df=reused_audit,
            regime_dist=reused_regime_dist, tz_table=tz_table,
            manifest=existing_manifest, summary=reused_summary,
            rollover_events=reused_events, rollover_ambiguous=reused_ambiguous,
            rollover_discarded=reused_discarded, segment_summary=reused_segments,
            parquet_path=parquet_path, parquet_sha256=sha256_file(parquet_path),
            audit_path=audit_path, regime_dist_path=regime_dist_path,
            manifest_path=manifest_path, summary_path=summary_path,
            tz_validation_path=tz_validation_path,
            rollover_events_path=rollover_events_path, rollover_ambiguous_path=rollover_ambiguous_path,
            rollover_discarded_path=rollover_discarded_path, segment_summary_path=segment_summary_path,
        )

    write_result = atomic_write_parquet(final_df, parquet_path)
    atomic_write_parquet(audit_df.reset_index(drop=True), audit_path)
    atomic_write_parquet(regime_dist.reset_index(drop=True), regime_dist_path)
    atomic_write_parquet(segment_summary.reset_index(drop=True), segment_summary_path)

    if not rollover_events.empty:
        atomic_write_parquet(rollover_events.reset_index(drop=True), rollover_events_path)
    else:
        rollover_events.to_parquet(rollover_events_path, index=False)
    if not rollover_ambiguous.empty:
        atomic_write_parquet(rollover_ambiguous.reset_index(drop=True), rollover_ambiguous_path)
    else:
        rollover_ambiguous.to_parquet(rollover_ambiguous_path, index=False)
    if not rollover_discarded.empty:
        atomic_write_parquet(rollover_discarded.reset_index(drop=True), rollover_discarded_path)
    else:
        rollover_discarded.to_parquet(rollover_discarded_path, index=False)

    summary = build_summary(manifest)
    write_json_atomic(manifest, manifest_path)
    write_json_atomic(summary, summary_path)
    write_json_atomic(tz_table.to_dict(orient="records"), tz_validation_path)

    return S01Result(
        reused_existing=False, df_window=final_df, audit_df=audit_df,
        regime_dist=regime_dist, tz_table=tz_table,
        manifest=manifest, summary=summary,
        rollover_events=rollover_events, rollover_ambiguous=rollover_ambiguous,
        rollover_discarded=rollover_discarded, segment_summary=segment_summary,
        parquet_path=parquet_path, parquet_sha256=write_result["sha256"],
        audit_path=audit_path, regime_dist_path=regime_dist_path,
        manifest_path=manifest_path, summary_path=summary_path,
        tz_validation_path=tz_validation_path,
        rollover_events_path=rollover_events_path, rollover_ambiguous_path=rollover_ambiguous_path,
        rollover_discarded_path=rollover_discarded_path, segment_summary_path=segment_summary_path,
    )
