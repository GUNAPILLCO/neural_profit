"""S02 v2 -- analisis intradia MNQ a partir de data/02_intraday/mnq_intraday_v2.parquet.

Reglas de alcance (ver revision tecnica y decisiones de reconstruccion de S02 v2):
- Parte exclusivamente de los artefactos aprobados de S01 v2
  (mnq_intraday_v2.parquet + trading_day_audit_v2.parquet). Nunca lee
  data/02_mnq_intraday/mnq_intraday.parquet (historico v1).
- No construye, evalua ni selecciona targets (DIR/BAR/OPC): eso es
  responsabilidad de S04 en adelante.
- No ejecuta ni reproduce el generador exploratorio de features de la
  notebook historica de S02 (queda como antecedente legacy, fuera de este
  modulo).
- "Cambio estructural" nunca se usa como conclusion: el diagnostico de
  inestabilidad temporal se limita a comparaciones exploratorias (rolling,
  zonas percentilicas), nunca a pruebas formales de ruptura.
- Toda ventana historica o futura exige simultaneamente: mismo `date`, mismo
  segmento de minutos consecutivos, mismo `contract` y consecutividad
  estricta de minuto a minuto. El cambio de contrato es bloqueante para la
  validez de la ventana (no solo diagnostico).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as _sps
from statsmodels.stats.diagnostic import het_arch

from src.data.s00_raw_ingestion import (
    IngestionError,
    atomic_write_parquet,
    get_git_provenance,
    normalized_config_bytes,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from src.data.s01_intraday_preparation import assign_consecutive_segments

MODULE_PATH = Path(__file__).resolve()

POPULATION_ALL = "all"
POPULATION_FULL_DAY = "full_day_eligible"
POPULATION_PARTIAL_REGIME = "partial_regime_eligible"
POPULATION_DESCRIPTIVE = "descriptive_only"

QUANTITATIVE_POPULATIONS = (POPULATION_FULL_DAY, POPULATION_PARTIAL_REGIME)


# ---------------------------------------------------------------------------
# 0) Config
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
# 1) Carga de artefactos de S01 v2 (solo lectura)
# ---------------------------------------------------------------------------

def load_s01_artifacts(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(project_root)
    intraday_path = project_root / config["input"]["intraday_parquet"]
    manifest_path = project_root / config["input"]["intraday_manifest"]
    audit_path = project_root / config["input"]["trading_day_audit"]

    for path in (intraday_path, manifest_path, audit_path):
        if not path.exists():
            raise IngestionError(
                f"Artefacto de S01 v2 no encontrado: {path}. "
                "S02 v2 no debe reconstruir ni asumir S01."
            )

    df_intraday = pd.read_parquet(intraday_path)
    df_audit = pd.read_parquet(audit_path)
    s01_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "df_intraday": df_intraday,
        "df_audit": df_audit,
        "s01_manifest": s01_manifest,
        "intraday_path": intraday_path,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
    }


# ---------------------------------------------------------------------------
# 2) Poblaciones
# ---------------------------------------------------------------------------

def attach_population_tags(df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    """Adjunta day_status/eligibility_category (por fecha) y
    regime_is_consecutive (por fecha+regime_id) a cada barra, sin recalcular
    nada que S01 v2 ya audito.

    Usa `.join()` (no `.merge()`) para preservar el DatetimeIndex original de
    `df` -- `.merge()` lo reemplaza silenciosamente por un RangeIndex, lo que
    romperia toda la logica temporal posterior (ordenamiento por indice,
    ventanas, ACF, etc.)."""
    out = df.copy()

    day_map = audit_df.set_index("date")[["day_status", "eligibility_category"]]
    out = out.join(day_map, on="date", how="left")
    if out["eligibility_category"].isna().any():
        raise IngestionError(
            "Existen barras sin eligibility_category tras el join con "
            "trading_day_audit_v2.parquet (fecha fuera del audit)."
        )

    regime_ids = sorted(
        int(c.split("_")[1])
        for c in audit_df.columns
        if c.startswith("regime_") and c.endswith("_is_consecutive")
    )
    if not regime_ids:
        raise IngestionError("trading_day_audit_v2.parquet no tiene columnas regime_*_is_consecutive.")

    long_frames = []
    for rid in regime_ids:
        col = f"regime_{rid}_is_consecutive"
        sub = audit_df[["date", col]].rename(columns={col: "regime_is_consecutive"})
        sub["regime_id"] = rid
        long_frames.append(sub)
    regime_map = pd.concat(long_frames, ignore_index=True).set_index(["date", "regime_id"])["regime_is_consecutive"]

    out = out.join(regime_map, on=["date", "regime_id"], how="left")
    if out["regime_is_consecutive"].isna().any():
        raise IngestionError(
            "Existen barras con regime_id sin entrada regime_{id}_is_consecutive "
            "en trading_day_audit_v2.parquet."
        )
    return out


def filter_population(df: pd.DataFrame, population: str) -> pd.DataFrame:
    """Filtra por poblacion segun las decisiones de reconstruccion de S02 v2.

    - all: sin filtrar (uso exclusivo: cobertura/calidad/descriptivo).
    - full_day_eligible: poblacion cuantitativa principal.
    - partial_regime_eligible: solo barras cuyo regimen esta marcado
      consecutivo por S01 (regime_is_consecutive == True), nunca el dia
      parcial completo.
    - descriptive_only: solo descripcion, nunca ventanas ni metricas para
      stages posteriores.
    not_model_eligible queda excluido de toda poblacion cuantitativa/
    descriptiva; solo aparece dentro de "all".
    """
    required = {"eligibility_category", "regime_is_consecutive"}
    missing = required - set(df.columns)
    if missing:
        raise IngestionError(f"Faltan columnas de poblacion (llamar attach_population_tags primero): {missing}")

    if population == POPULATION_ALL:
        return df.copy()
    if population == POPULATION_FULL_DAY:
        return df.loc[df["eligibility_category"] == POPULATION_FULL_DAY].copy()
    if population == POPULATION_PARTIAL_REGIME:
        mask = (df["eligibility_category"] == POPULATION_PARTIAL_REGIME) & (df["regime_is_consecutive"].astype(bool))
        return df.loc[mask].copy()
    if population == POPULATION_DESCRIPTIVE:
        return df.loc[df["eligibility_category"] == POPULATION_DESCRIPTIVE].copy()
    raise ValueError(f"Poblacion desconocida: {population!r}")


# ---------------------------------------------------------------------------
# 3) Validacion temporal (gap-aware, escapada por consecutive_segment_id)
# ---------------------------------------------------------------------------

def validate_intraday_dataset(df: pd.DataFrame) -> dict[str, Any]:
    required_cols = [
        "date", "minute_of_day", "regime_id", "consecutive_segment_id",
        "open", "high", "low", "close", "volume", "contract",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise IngestionError(f"Faltan columnas requeridas para validar: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IngestionError("El indice de mnq_intraday debe ser un DatetimeIndex.")
    if df.empty:
        raise IngestionError("El dataset esta vacio.")

    checks: dict[str, bool] = {}
    summary: dict[str, Any] = {}

    checks["index_is_monotonic_increasing"] = bool(df.index.is_monotonic_increasing)
    checks["index_is_unique"] = bool(df.index.is_unique)

    index_dates = df.index.tz_localize(None).date if df.index.tz is not None else df.index.date
    col_dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
    checks["date_column_matches_index_date"] = bool((pd.Index(index_dates) == pd.Index(col_dates)).all())

    diffs_sec = pd.Series(df.index).diff().dt.total_seconds()
    n_non_positive = int((diffs_sec.iloc[1:] <= 0).sum())
    checks["all_global_time_diffs_positive"] = n_non_positive == 0
    summary["n_non_positive_global_diffs"] = n_non_positive

    # Consecutividad estricta DENTRO de cada consecutive_segment_id: debe
    # cumplirse por construccion (S01 v2); se verifica aqui de forma
    # defensiva, no se corrige silenciosamente.
    seg_mod_diff = df.groupby("consecutive_segment_id", sort=False)["minute_of_day"].diff()
    seg_bad_steps = int(((seg_mod_diff.notna()) & (seg_mod_diff != 1)).sum())
    checks["segments_are_strictly_consecutive_minutes"] = seg_bad_steps == 0
    summary["n_bad_steps_within_segment"] = seg_bad_steps

    dup_minute_within_day = int(df.duplicated(subset=["date", "minute_of_day"]).sum())
    checks["no_duplicate_minute_of_day_within_day"] = dup_minute_within_day == 0
    summary["n_duplicate_minute_of_day_within_day"] = dup_minute_within_day

    # Gaps intradia dentro de una misma fecha: INFORMATIVO, no critico. v2
    # incluye jornadas parciales con gaps reales y documentados por S01; no
    # es un error, es la razon de ser de consecutive_segment_id.
    day_gap_diff = df.groupby("date", sort=False)["minute_of_day"].diff()
    n_intraday_gaps = int(((day_gap_diff.notna()) & (day_gap_diff != 1)).sum())
    summary["n_intraday_gaps_within_date_informational"] = n_intraday_gaps

    summary["n_rows"] = int(len(df))
    summary["n_dates"] = int(df["date"].nunique())
    summary["n_segments"] = int(df["consecutive_segment_id"].nunique())
    summary["start"] = str(df.index.min())
    summary["end"] = str(df.index.max())

    critical_checks = [
        "index_is_monotonic_increasing",
        "index_is_unique",
        "date_column_matches_index_date",
        "all_global_time_diffs_positive",
        "segments_are_strictly_consecutive_minutes",
        "no_duplicate_minute_of_day_within_day",
    ]
    failed = [k for k in critical_checks if not checks.get(k, False)]

    return {"ok": len(failed) == 0, "checks": checks, "summary": summary, "failed_critical_checks": failed}


# ---------------------------------------------------------------------------
# 4) Metricas OHLCV descriptivas (causales, escapadas por segmento)
# ---------------------------------------------------------------------------

def build_ohlcv_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = ["consecutive_segment_id", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise IngestionError(f"Faltan columnas para metricas OHLCV: {missing}")

    out = df.sort_index().copy()
    grp = out.groupby("consecutive_segment_id", sort=False)

    out["prev_close_1m"] = grp["close"].shift(1)
    out["delta_close_1m"] = out["close"] - out["prev_close_1m"]
    out["ret_1m"] = out["close"] / out["prev_close_1m"] - 1
    out["log_ret_1m"] = np.log(out["close"] / out["prev_close_1m"])
    out["abs_delta_close_1m"] = out["delta_close_1m"].abs()
    out["abs_ret_1m"] = out["ret_1m"].abs()
    out["squared_ret_1m"] = out["ret_1m"] ** 2

    out["range_pts"] = out["high"] - out["low"]
    out["range_pct"] = out["range_pts"] / out["close"]

    # body_signed_pts / body_abs_pts: nombres inequivocos (ver S02-05 en
    # 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md). Nunca usar "body_pts" a secas.
    out["body_signed_pts"] = out["close"] - out["open"]
    out["body_abs_pts"] = out["body_signed_pts"].abs()

    out["candle_direction"] = np.select(
        [out["close"] > out["open"], out["close"] < out["open"]],
        [1, -1],
        default=0,
    )

    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    out["upper_wick_pts"] = (out["high"] - body_high).clip(lower=0)
    out["lower_wick_pts"] = (body_low - out["low"]).clip(lower=0)

    out["close_position"] = np.where(
        out["range_pts"] > 0, (out["close"] - out["low"]) / out["range_pts"], np.nan
    )
    out["body_to_range"] = np.where(
        out["range_pts"] > 0, out["body_abs_pts"] / out["range_pts"], np.nan
    )

    out["log_volume"] = np.log1p(out["volume"].clip(lower=0))

    return out


# ---------------------------------------------------------------------------
# 5) Ventanas operativas 30/60/90 -- 4 condiciones simultaneas
# ---------------------------------------------------------------------------

def assign_window_segment_id(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula un segmento de minutos consecutivos SOBRE LA POBLACION YA
    FILTRADA. No reutiliza consecutive_segment_id de S01 directamente porque,
    para partial_regime_eligible, una barra retenida puede pertenecer a un
    segmento original de S01 que ya no es enteramente contiguo tras el
    filtrado (p. ej. si solo un regimen dentro del segmento es elegible).
    Recalcular con la misma logica de S01 (assign_consecutive_segments) es la
    forma correcta de garantizar "minutos consecutivos" sobre la poblacion
    efectivamente usada, no solo sobre el dataset completo."""
    out = df.copy()
    out["window_segment_id"] = assign_consecutive_segments(out)
    return out


def _build_contract_run_id(df: pd.DataFrame) -> np.ndarray:
    contract_arr = df["contract"].to_numpy()
    changed = np.empty(len(df), dtype=bool)
    if len(df) > 0:
        changed[0] = True
        if len(df) > 1:
            changed[1:] = contract_arr[1:] != contract_arr[:-1]
    return np.cumsum(changed)


def build_window_validity(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Marca la validez de la ventana historica/futura para cada horizonte.

    Una ventana es valida si y solo si, simultaneamente:
    1) pertenece a la misma fecha (implicito en window_segment_id);
    2) pertenece al mismo window_segment_id (tramo de minutos consecutivos
       sobre la poblacion filtrada);
    3) pertenece a un unico contrato;
    4) tiene una secuencia completa de minutos consecutivos (garantizado por
       construccion de window_segment_id, verificado en validate_intraday_dataset
       de forma defensiva a nivel de todo el dataset, no por ventana).

    El motivo de invalidez se etiqueta explicitamente como
    "insufficient_bars" o "contract_change" para poder auditar el impacto de
    rollover por separado (ver build_rollover_window_audit).
    """
    required = ["date", "minute_of_day", "contract"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise IngestionError(f"Faltan columnas para ventanas: {missing}")

    out = df.sort_index().copy()
    out = assign_window_segment_id(out)
    out["contract_run_id"] = _build_contract_run_id(out)

    seg_grp = out.groupby("window_segment_id", sort=False)
    out["seg_bar_pos"] = seg_grp.cumcount()
    out["seg_bars_total"] = seg_grp["window_segment_id"].transform("size")

    for h in horizons:
        hist_bars_ok = out["seg_bar_pos"] >= (h - 1)
        future_bars_ok = out["seg_bar_pos"] <= (out["seg_bars_total"] - h - 1)

        # contract_run_id es monotono y nunca repite un valor previo: si
        # coincide en ambos extremos de la ventana, todo el tramo comparte
        # contrato.
        run_at_hist_start = out.groupby("window_segment_id", sort=False)["contract_run_id"].shift(h - 1)
        hist_contract_ok = (run_at_hist_start == out["contract_run_id"]).fillna(False)

        run_at_future_end = out.groupby("window_segment_id", sort=False)["contract_run_id"].shift(-h)
        future_contract_ok = (run_at_future_end == out["contract_run_id"]).fillna(False)

        hist_valid = hist_bars_ok & hist_contract_ok
        future_valid = future_bars_ok & future_contract_ok

        out[f"hist_window_valid_{h}m"] = hist_valid
        out[f"future_window_valid_{h}m"] = future_valid
        out[f"full_window_valid_{h}m"] = hist_valid & future_valid

        out[f"hist_invalid_reason_{h}m"] = np.where(
            hist_valid, "valid", np.where(~hist_bars_ok, "insufficient_bars", "contract_change")
        )
        out[f"future_invalid_reason_{h}m"] = np.where(
            future_valid, "valid", np.where(~future_bars_ok, "insufficient_bars", "contract_change")
        )

    return out


def _rolling_by_segment(df: pd.DataFrame, col: str, window: int, agg: str, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    return (
        df.groupby("window_segment_id", sort=False)[col]
        .rolling(window=window, min_periods=min_periods)
        .agg(agg)
        .reset_index(level=0, drop=True)
    )


def _future_rolling_by_segment(df: pd.DataFrame, col: str, window: int, agg: str, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window

    def _future_roll(s: pd.Series) -> pd.Series:
        shifted = s.shift(-1)
        reversed_shifted = shifted.iloc[::-1]
        rolled = reversed_shifted.rolling(window=window, min_periods=min_periods).agg(agg)
        return rolled.iloc[::-1]

    return df.groupby("window_segment_id", sort=False)[col].transform(_future_roll)


def build_historical_window_metrics(df_valid: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df_valid.copy()
    for h in horizons:
        valid_col = f"hist_window_valid_{h}m"
        created: list[str] = []

        first_close = out.groupby("window_segment_id", sort=False)["close"].shift(h - 1)
        out[f"hist_first_close_{h}m"] = first_close
        out[f"hist_close_return_pts_{h}m"] = out["close"] - first_close
        out[f"hist_close_return_pct_{h}m"] = out["close"] / first_close - 1
        out[f"hist_log_return_{h}m"] = np.log(out["close"] / first_close)
        out[f"hist_abs_return_pts_{h}m"] = out[f"hist_close_return_pts_{h}m"].abs()
        created += [
            f"hist_first_close_{h}m", f"hist_close_return_pts_{h}m", f"hist_close_return_pct_{h}m",
            f"hist_log_return_{h}m", f"hist_abs_return_pts_{h}m",
        ]

        high_max = _rolling_by_segment(out, "high", h, "max")
        low_min = _rolling_by_segment(out, "low", h, "min")
        out[f"hist_high_max_{h}m"] = high_max
        out[f"hist_low_min_{h}m"] = low_min
        out[f"hist_range_pts_{h}m"] = high_max - low_min
        out[f"hist_range_pct_{h}m"] = out[f"hist_range_pts_{h}m"] / out["close"]
        created += [f"hist_high_max_{h}m", f"hist_low_min_{h}m", f"hist_range_pts_{h}m", f"hist_range_pct_{h}m"]

        close_max = _rolling_by_segment(out, "close", h, "max")
        close_min = _rolling_by_segment(out, "close", h, "min")
        out[f"hist_close_max_{h}m"] = close_max
        out[f"hist_close_min_{h}m"] = close_min
        out[f"hist_close_range_pts_{h}m"] = close_max - close_min
        created += [f"hist_close_max_{h}m", f"hist_close_min_{h}m", f"hist_close_range_pts_{h}m"]

        out[f"hist_realized_vol_{h}m"] = _rolling_by_segment(out, "log_ret_1m", h - 1, "std", min_periods=h - 1)
        created.append(f"hist_realized_vol_{h}m")

        out[f"hist_volume_sum_{h}m"] = _rolling_by_segment(out, "volume", h, "sum")
        out[f"hist_volume_mean_{h}m"] = _rolling_by_segment(out, "volume", h, "mean")
        out[f"hist_volume_max_{h}m"] = _rolling_by_segment(out, "volume", h, "max")
        out[f"hist_log_volume_mean_{h}m"] = _rolling_by_segment(out, "log_volume", h, "mean")
        created += [f"hist_volume_sum_{h}m", f"hist_volume_mean_{h}m", f"hist_volume_max_{h}m", f"hist_log_volume_mean_{h}m"]

        out.loc[~out[valid_col], created] = np.nan
    return out


def build_future_window_metrics(df_valid: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df_valid.copy()
    for h in horizons:
        valid_col = f"future_window_valid_{h}m"
        created: list[str] = []

        future_close = out.groupby("window_segment_id", sort=False)["close"].shift(-h)
        out[f"future_close_at_{h}m"] = future_close
        out[f"future_return_pts_{h}m"] = future_close - out["close"]
        out[f"future_return_pct_{h}m"] = future_close / out["close"] - 1
        out[f"future_log_return_{h}m"] = np.log(future_close / out["close"])
        out[f"future_abs_return_pts_{h}m"] = out[f"future_return_pts_{h}m"].abs()
        created += [
            f"future_close_at_{h}m", f"future_return_pts_{h}m", f"future_return_pct_{h}m",
            f"future_log_return_{h}m", f"future_abs_return_pts_{h}m",
        ]

        future_high_max = _future_rolling_by_segment(out, "high", h, "max")
        future_low_min = _future_rolling_by_segment(out, "low", h, "min")
        out[f"future_high_max_{h}m"] = future_high_max
        out[f"future_low_min_{h}m"] = future_low_min
        out[f"future_range_pts_{h}m"] = future_high_max - future_low_min
        out[f"future_range_pct_{h}m"] = out[f"future_range_pts_{h}m"] / out["close"]
        created += [f"future_high_max_{h}m", f"future_low_min_{h}m", f"future_range_pts_{h}m", f"future_range_pct_{h}m"]

        future_close_max = _future_rolling_by_segment(out, "close", h, "max")
        future_close_min = _future_rolling_by_segment(out, "close", h, "min")
        out[f"future_close_max_{h}m"] = future_close_max
        out[f"future_close_min_{h}m"] = future_close_min
        out[f"future_close_range_pts_{h}m"] = future_close_max - future_close_min
        created += [f"future_close_max_{h}m", f"future_close_min_{h}m", f"future_close_range_pts_{h}m"]

        out[f"future_up_excursion_pts_{h}m"] = (future_high_max - out["close"]).clip(lower=0)
        out[f"future_down_excursion_pts_{h}m"] = (out["close"] - future_low_min).clip(lower=0)
        out[f"future_max_excursion_pts_{h}m"] = out[[f"future_up_excursion_pts_{h}m", f"future_down_excursion_pts_{h}m"]].max(axis=1)
        created += [f"future_up_excursion_pts_{h}m", f"future_down_excursion_pts_{h}m", f"future_max_excursion_pts_{h}m"]

        out[f"future_realized_vol_{h}m"] = _future_rolling_by_segment(out, "log_ret_1m", h, "std", min_periods=h)
        created.append(f"future_realized_vol_{h}m")

        out[f"future_volume_sum_{h}m"] = _future_rolling_by_segment(out, "volume", h, "sum")
        out[f"future_volume_mean_{h}m"] = _future_rolling_by_segment(out, "volume", h, "mean")
        out[f"future_volume_max_{h}m"] = _future_rolling_by_segment(out, "volume", h, "max")
        out[f"future_log_volume_mean_{h}m"] = _future_rolling_by_segment(out, "log_volume", h, "mean")
        created += [f"future_volume_sum_{h}m", f"future_volume_mean_{h}m", f"future_volume_max_{h}m", f"future_log_volume_mean_{h}m"]

        out.loc[~out[valid_col], created] = np.nan
    return out


def build_window_validity_summary(df_valid: pd.DataFrame, horizons: list[int], population: str) -> pd.DataFrame:
    rows = []
    total = len(df_valid)
    for h in horizons:
        for window_type, valid_col, reason_col in (
            ("historical", f"hist_window_valid_{h}m", f"hist_invalid_reason_{h}m"),
            ("future", f"future_window_valid_{h}m", f"future_invalid_reason_{h}m"),
        ):
            valid_rows = int(df_valid[valid_col].sum())
            reasons = df_valid.loc[~df_valid[valid_col], reason_col].value_counts()
            rows.append({
                "population": population,
                "horizon_minutes": h,
                "window_type": window_type,
                "total_rows": total,
                "valid_rows": valid_rows,
                "invalid_rows": total - valid_rows,
                "invalid_insufficient_bars": int(reasons.get("insufficient_bars", 0)),
                "invalid_contract_change": int(reasons.get("contract_change", 0)),
                "pct_valid": (valid_rows / total * 100.0) if total else np.nan,
            })
        full_valid = int(df_valid[f"full_window_valid_{h}m"].sum())
        rows.append({
            "population": population,
            "horizon_minutes": h,
            "window_type": "full",
            "total_rows": total,
            "valid_rows": full_valid,
            "invalid_rows": total - full_valid,
            "invalid_insufficient_bars": np.nan,
            "invalid_contract_change": np.nan,
            "pct_valid": (full_valid / total * 100.0) if total else np.nan,
        })
    return pd.DataFrame(rows)


def build_rollover_window_audit(df_valid: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Resumen de ventanas invalidadas especificamente por cambio de
    contrato (no duplica la auditoria general de rollover de la Fase 4 del
    rebuild plan, que audita las transiciones crudas; esto cuantifica el
    impacto especifico a nivel de ventana historica/futura)."""
    out = df_valid.sort_index()
    contract_arr = out["contract"].to_numpy()
    seg_arr = out["window_segment_id"].to_numpy()
    ts_arr = out.index.to_numpy()

    transition_positions = []
    for i in range(1, len(out)):
        if contract_arr[i] != contract_arr[i - 1]:
            transition_positions.append(i)

    rows = []
    for transition_id, pos in enumerate(transition_positions):
        same_segment = seg_arr[pos] == seg_arr[pos - 1]
        contract_before = contract_arr[pos - 1]
        contract_after = contract_arr[pos]
        ts_before = pd.Timestamp(ts_arr[pos - 1])
        ts_after = pd.Timestamp(ts_arr[pos])
        gap_minutes = (ts_after - ts_before).total_seconds() / 60.0

        for h in horizons:
            n_hist = n_future = 0
            if same_segment:
                seg_id = seg_arr[pos]
                seg_mask = seg_arr == seg_id
                pos_in_seg = out["seg_bar_pos"].to_numpy()
                p_pos = pos_in_seg[pos]

                hist_reason = out[f"hist_invalid_reason_{h}m"].to_numpy()
                hist_affected = (
                    seg_mask
                    & (pos_in_seg >= p_pos)
                    & (pos_in_seg <= p_pos + h - 1)
                    & (hist_reason == "contract_change")
                )
                n_hist = int(hist_affected.sum())

                future_reason = out[f"future_invalid_reason_{h}m"].to_numpy()
                future_affected = (
                    seg_mask
                    & (pos_in_seg <= p_pos - 1)
                    & (pos_in_seg >= p_pos - h)
                    & (future_reason == "contract_change")
                )
                n_future = int(future_affected.sum())

            rows.append({
                "transition_id": transition_id,
                "contract_before": contract_before,
                "contract_after": contract_after,
                "last_bar_before_ts": str(ts_before),
                "first_bar_after_ts": str(ts_after),
                "gap_minutes_between_contracts": gap_minutes,
                "intra_segment_transition": bool(same_segment),
                "horizon_minutes": h,
                "n_bars_hist_invalidated_by_contract": n_hist,
                "n_bars_future_invalidated_by_contract": n_future,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6) Resumenes descriptivos (cobertura, regimen, estadistica OHLCV)
# ---------------------------------------------------------------------------

def build_coverage_summary(df_all_tagged: pd.DataFrame) -> pd.DataFrame:
    """Cobertura sobre TODOS los datos observados (poblacion 'all'), por
    year/quarter/month/contract/day_status/eligibility_category."""
    df = df_all_tagged.copy()
    df["year"] = df.index.year
    df["quarter"] = df.index.year.astype(str) + "Q" + df.index.quarter.astype(str)
    df["month"] = df.index.strftime("%Y-%m")

    total_rows = len(df)
    rows = []
    cut_dims = {
        "year": "year",
        "quarter": "quarter",
        "month": "month",
        "contract": "contract",
        "day_status": "day_status",
        "eligibility_category": "eligibility_category",
    }
    for cut_dimension, col in cut_dims.items():
        agg = df.groupby(col, observed=True).agg(
            n_rows=("close", "size"),
            n_dates=("date", "nunique"),
        ).reset_index().rename(columns={col: "cut_value"})
        # cut_value combina tipos heterogeneos entre cortes (year: int,
        # quarter/month/contract/day_status/eligibility: str). Se
        # normaliza a str para poder concatenar y persistir en Parquet sin
        # ambiguedad de tipo de columna.
        agg["cut_value"] = agg["cut_value"].astype(str)
        agg["cut_dimension"] = cut_dimension
        agg["avg_bars_per_day_observed"] = agg["n_rows"] / agg["n_dates"]
        agg["pct_of_total_rows"] = agg["n_rows"] / total_rows * 100.0 if total_rows else np.nan
        rows.append(agg[["cut_dimension", "cut_value", "n_rows", "n_dates", "avg_bars_per_day_observed", "pct_of_total_rows"]])

    return pd.concat(rows, ignore_index=True)


def build_regime_distribution_summary(df_by_population: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Distribucion de regimen por poblacion x corte (overall/year/contract)."""
    frames = []
    for population, df in df_by_population.items():
        if df.empty:
            continue
        df = df.copy()
        df["year"] = df.index.year

        for cut_dimension, col in (("overall", None), ("year", "year"), ("contract", "contract")):
            if col is None:
                grp_cols = ["regime_id", "regime_label"]
                base = df.assign(_cut="overall")
                grouped = base.groupby(["_cut"] + grp_cols, observed=True).agg(
                    n_bars=("close", "size"), n_days=("date", "nunique"),
                ).reset_index().rename(columns={"_cut": "cut_value"})
            else:
                grouped = df.groupby([col, "regime_id", "regime_label"], observed=True).agg(
                    n_bars=("close", "size"), n_days=("date", "nunique"),
                ).reset_index().rename(columns={col: "cut_value"})

            # cut_value combina int (year) y str (contract/"overall") entre
            # cortes; normalizar a str antes de concatenar/persistir.
            grouped["cut_value"] = grouped["cut_value"].astype(str)
            total_within_cut = grouped.groupby("cut_value")["n_bars"].transform("sum")
            grouped["pct_within_cut"] = grouped["n_bars"] / total_within_cut * 100.0
            grouped["cut_dimension"] = cut_dimension
            grouped["population"] = population
            frames.append(grouped[["population", "cut_dimension", "cut_value", "regime_id", "regime_label", "n_bars", "n_days", "pct_within_cut"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["population", "cut_dimension", "cut_value", "regime_id", "regime_label", "n_bars", "n_days", "pct_within_cut"]
    )


_PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_OHLCV_STAT_METRICS = [
    "delta_close_1m", "ret_1m", "log_ret_1m", "abs_delta_close_1m", "abs_ret_1m",
    "range_pts", "range_pct", "body_signed_pts", "body_abs_pts",
    "upper_wick_pts", "lower_wick_pts", "close_position", "body_to_range", "log_volume",
]


def _describe_series(s: pd.Series) -> dict[str, Any]:
    clean = s.replace([np.inf, -np.inf], np.nan)
    valid = clean.dropna()
    n_inf = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum())
    result = {
        "count": int(valid.shape[0]),
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "std": float(valid.std()) if len(valid) else np.nan,
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
        "n_inf": n_inf,
        "n_nan": int(s.isna().sum()),
    }
    for p in _PERCENTILES:
        result[f"p{int(p * 100):02d}"] = float(valid.quantile(p)) if len(valid) else np.nan
    return result


def build_ohlcv_stats_summary(df_by_population: dict[str, pd.DataFrame], metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or _OHLCV_STAT_METRICS
    rows = []
    for population, df in df_by_population.items():
        if df.empty:
            continue
        df = df.copy()
        df["year"] = df.index.year
        df["quarter"] = df.index.year.astype(str) + "Q" + df.index.quarter.astype(str)

        cut_specs: list[tuple[str, pd.Series | None]] = [("overall", None)]
        cut_specs.append(("year", df["year"]))
        cut_specs.append(("quarter", df["quarter"]))
        if "contract" in df.columns:
            cut_specs.append(("contract", df["contract"]))
        if "regime_label" in df.columns:
            cut_specs.append(("regime", df["regime_label"]))

        for cut_dimension, cut_series in cut_specs:
            if cut_series is None:
                groups = {"overall": df}
            else:
                groups = {str(k): g for k, g in df.groupby(cut_series, observed=True)}
            for cut_value, g in groups.items():
                for metric in metrics:
                    if metric not in g.columns:
                        continue
                    stats = _describe_series(g[metric])
                    stats.update({
                        "population": population, "cut_dimension": cut_dimension,
                        "cut_value": cut_value, "metric": metric,
                    })
                    rows.append(stats)
    return pd.DataFrame(rows)


def build_ohlcv_correlation(df: pd.DataFrame, population: str, metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or _OHLCV_STAT_METRICS
    available = [m for m in metrics if m in df.columns]
    clean = df[available].replace([np.inf, -np.inf], np.nan)

    rows = []
    for method in ("pearson", "spearman"):
        corr = clean.corr(method=method)
        for i, a in enumerate(available):
            for b in available[i + 1:]:
                rows.append({
                    "population": population, "metric_a": a, "metric_b": b,
                    "method": method, "correlation": float(corr.loc[a, b]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7) Resumen agregado de metricas por ventana (reemplaza el parquet
#    masivo por-barra; solo se persisten estadisticos agregados)
# ---------------------------------------------------------------------------

_WINDOW_METRIC_BASES = [
    "close_return_pts", "abs_return_pts", "range_pts", "close_range_pts", "realized_vol", "volume_sum",
]
_FUTURE_ONLY_BASES = ["up_excursion_pts", "down_excursion_pts", "max_excursion_pts"]


def build_window_metrics_summary(df_metrics: pd.DataFrame, horizons: list[int], population: str) -> pd.DataFrame:
    df = df_metrics.copy()
    df["year"] = df.index.year
    df["quarter"] = df.index.year.astype(str) + "Q" + df.index.quarter.astype(str)

    cut_specs: list[tuple[str, pd.Series | None]] = [("overall", None), ("year", df["year"]), ("quarter", df["quarter"])]
    if "contract" in df.columns:
        cut_specs.append(("contract", df["contract"]))
    if "regime_label" in df.columns:
        cut_specs.append(("regime", df["regime_label"]))

    rows = []
    for h in horizons:
        for window_type, valid_col, bases in (
            ("historical", f"hist_window_valid_{h}m", [f"hist_{b}_{h}m" for b in _WINDOW_METRIC_BASES]),
            ("future", f"future_window_valid_{h}m", [f"future_{b}_{h}m" for b in _WINDOW_METRIC_BASES + _FUTURE_ONLY_BASES]),
        ):
            valid_df = df.loc[df[valid_col]]
            for metric_col in bases:
                if metric_col not in df.columns:
                    continue
                metric_base = metric_col.replace(f"hist_", "").replace(f"future_", "").replace(f"_{h}m", "")
                for cut_dimension, cut_series in cut_specs:
                    if cut_series is None:
                        groups = {"overall": valid_df}
                    else:
                        groups = {str(k): g for k, g in valid_df.groupby(cut_series.loc[valid_df.index], observed=True)}
                    for cut_value, g in groups.items():
                        s = g[metric_col].replace([np.inf, -np.inf], np.nan).dropna()
                        if s.empty:
                            continue
                        rows.append({
                            "population": population, "horizon_minutes": h, "window_type": window_type,
                            "metric_base": metric_base, "cut_dimension": cut_dimension, "cut_value": cut_value,
                            "count": int(len(s)), "mean": float(s.mean()),
                            "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.50)),
                            "p75": float(s.quantile(0.75)), "p90": float(s.quantile(0.90)),
                            "p95": float(s.quantile(0.95)), "p99": float(s.quantile(0.99)),
                            "max": float(s.max()),
                        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8) Estabilidad (CV), rolling y diagnostico de inestabilidad temporal
#    (nunca "cambio estructural")
# ---------------------------------------------------------------------------

def _classify_cv(cv: float) -> str:
    if pd.isna(cv):
        return "sin datos"
    if cv < 0.15:
        return "estable"
    if cv < 0.30:
        return "moderada"
    return "inestable"


def build_stability_summary(df_window_metrics_summary: pd.DataFrame, population: str) -> pd.DataFrame:
    """CV entre cortes (year/quarter/contract/regime) para cada
    (horizonte, metric_base), a partir del resumen agregado por corte (no
    del per-barra)."""
    df = df_window_metrics_summary.loc[df_window_metrics_summary["population"] == population]
    df = df.loc[df["cut_dimension"] != "overall"]

    rows = []
    for (horizon, window_type, metric_base, cut_dimension), g in df.groupby(
        ["horizon_minutes", "window_type", "metric_base", "cut_dimension"], observed=True
    ):
        if g.empty:
            continue
        p50_mean, p50_std = g["p50"].mean(), g["p50"].std()
        p95_mean, p95_std = g["p95"].mean(), g["p95"].std()
        p50_cv = p50_std / p50_mean if p50_mean else np.nan
        p95_cv = p95_std / p95_mean if p95_mean else np.nan
        top_p50 = g.sort_values("p50", ascending=False).iloc[0]
        low_p50 = g.sort_values("p50", ascending=True).iloc[0]
        rows.append({
            "population": population, "context_name": cut_dimension, "horizon_minutes": horizon,
            "window_type": window_type, "metric_base": metric_base, "n_groups": int(len(g)),
            "p50_min": float(g["p50"].min()), "p50_max": float(g["p50"].max()),
            "p50_cv": float(p50_cv) if pd.notna(p50_cv) else np.nan,
            "p50_stability": _classify_cv(p50_cv),
            "top_p50_group": str(top_p50["cut_value"]), "low_p50_group": str(low_p50["cut_value"]),
            "p95_cv": float(p95_cv) if pd.notna(p95_cv) else np.nan,
            "p95_stability": _classify_cv(p95_cv),
        })
    return pd.DataFrame(rows)


def build_daily_aggregates(df_metrics: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df_metrics.copy()
    df["year"] = df.index.year
    df["year_quarter"] = df.index.year.astype(str) + "Q" + df.index.quarter.astype(str)
    df["_ts"] = df.index

    agg_dict = {
        "rows": ("close", "count"),
        "first_timestamp": ("_ts", "min"),
        "last_timestamp": ("_ts", "max"),
        "year": ("year", "first"),
        "year_quarter": ("year_quarter", "first"),
    }
    daily = df.groupby("date", observed=True).agg(**agg_dict).reset_index()
    daily["date_dt"] = pd.to_datetime(daily["date"])

    for h in horizons:
        valid_col = f"future_window_valid_{h}m"
        metric_col = f"future_max_excursion_pts_{h}m"
        if valid_col not in df.columns or metric_col not in df.columns:
            continue
        sub = df.loc[df[valid_col], ["date", metric_col]].replace([np.inf, -np.inf], np.nan).dropna()
        daily_metric = sub.groupby("date", observed=True)[metric_col].agg(
            p50=lambda x: x.quantile(0.50), p95=lambda x: x.quantile(0.95),
        ).reset_index()
        daily_metric = daily_metric.rename(columns={
            "p50": f"future_max_excursion_pts_{h}m_daily_p50",
            "p95": f"future_max_excursion_pts_{h}m_daily_p95",
        })
        daily = daily.merge(daily_metric, on="date", how="left")

    return daily.sort_values("date_dt").reset_index(drop=True)


def build_rolling_summary(daily_df: pd.DataFrame, horizons: list[int], rolling_windows_days: list[int], min_period_ratio: float) -> pd.DataFrame:
    daily = daily_df.sort_values("date_dt").set_index("date_dt")
    rows = []
    for window in rolling_windows_days:
        min_periods = max(5, int(window * min_period_ratio))
        for h in horizons:
            for stat in ("p50", "p95"):
                source_col = f"future_max_excursion_pts_{h}m_daily_{stat}"
                if source_col not in daily.columns:
                    continue
                rolled = daily[source_col].rolling(window=window, min_periods=min_periods).median()
                series = rolled.dropna()
                if series.empty:
                    continue
                rows.append({
                    "rolling_window_days": window, "horizon_minutes": h, "metric_base": "future_max_excursion_pts", "stat": stat,
                    "first_valid_date": str(series.index.min()), "last_valid_date": str(series.index.max()),
                    "latest_value": float(series.iloc[-1]), "min_value": float(series.min()), "max_value": float(series.max()),
                    "mean_value": float(series.mean()), "median_value": float(series.median()),
                })
    return pd.DataFrame(rows)


def build_temporal_instability_zones(daily_df: pd.DataFrame, horizons: list[int], rolling_windows_days: list[int], min_period_ratio: float, zone_percentiles: dict[str, float]) -> pd.DataFrame:
    """Diagnostico exploratorio de inestabilidad temporal / cambio de
    distribucion (NUNCA "cambio estructural"): marca dias donde el rolling de
    la mediana diaria de future_max_excursion_pts esta en zona alta/extrema
    respecto de su propio historico."""
    daily = daily_df.sort_values("date_dt").set_index("date_dt")
    rows = []
    for window in rolling_windows_days:
        min_periods = max(5, int(window * min_period_ratio))
        for h in horizons:
            source_col = f"future_max_excursion_pts_{h}m_daily_p50"
            if source_col not in daily.columns:
                continue
            rolling_col = daily[source_col].rolling(window=window, min_periods=min_periods).median()
            series = pd.DataFrame({
                "value": rolling_col,
                "year": daily["year"],
                "year_quarter": daily["year_quarter"],
            }).dropna()
            if series.empty:
                continue
            p_high = series["value"].quantile(zone_percentiles["high"])
            p_extreme = series["value"].quantile(zone_percentiles["extreme"])
            series["zone"] = np.select(
                [series["value"] >= p_extreme, series["value"] >= p_high],
                ["extremo", "alto"], default="normal",
            )
            zone_summary = series.groupby(["year", "year_quarter", "zone"], observed=True).agg(
                days=("value", "count"), value_median=("value", "median"), value_max=("value", "max"),
            ).reset_index()
            zone_summary["rolling_window_days"] = window
            zone_summary["horizon_minutes"] = h
            zone_summary["p_high_threshold"] = p_high
            zone_summary["p_extreme_threshold"] = p_extreme
            rows.append(zone_summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["year", "year_quarter", "zone", "days", "value_median", "value_max", "rolling_window_days", "horizon_minutes", "p_high_threshold", "p_extreme_threshold"]
    )


# ---------------------------------------------------------------------------
# 9) Dependencia temporal: ACF, Ljung-Box, ARCH-LM (diagnostico, no
#    selecciona features)
# ---------------------------------------------------------------------------

def _clean_series_with_segment(sub: pd.DataFrame, series_name: str, segment_col: str = "consecutive_segment_id") -> tuple[np.ndarray, np.ndarray]:
    """Limpia inf/NaN preservando el segmento de origen de cada observacion
    superviviente, sin alterar el orden cronologico. Necesario para que la
    ACF/Ljung-Box gap-aware sepan, tras el dropna, que dos posiciones
    consecutivas en el array pueden pertenecer a segmentos distintos (p. ej.
    la ultima barra de un dia y la segunda barra -- la primera es NaN por el
    propio calculo de retorno -- del dia siguiente)."""
    tmp = sub[[series_name, segment_col]].replace([np.inf, -np.inf], np.nan).dropna(subset=[series_name])
    return tmp[series_name].to_numpy(dtype=float), tmp[segment_col].to_numpy()


def _segment_blocks(values: np.ndarray, segments: np.ndarray) -> list[np.ndarray]:
    """Parte `values` (ya en orden cronologico) en bloques contiguos que
    comparten el mismo id de segmento, preservando el orden. Como cada
    segmento solo pierde su primera barra en el dropna (unico NaN posible en
    ret_1m/abs_ret_1m/squared_ret_1m), lo que resta de cada segmento sigue
    siendo un bloque contiguo dentro de `values`."""
    if len(values) == 0:
        return []
    change_points = np.where(segments[1:] != segments[:-1])[0] + 1
    return np.split(values, change_points)


def _gap_aware_acf(values: np.ndarray, segments: np.ndarray, max_lag: int) -> dict[str, Any]:
    """ACF que NUNCA forma un par (x_t, x_{t+k}) entre dos observaciones de
    distinto `consecutive_segment_id`. La media y la varianza global (c0) se
    calculan sobre TODAS las observaciones validas (son reductores escalares,
    no pares); solo los productos cruzados numerador se acumulan
    exclusivamente dentro de cada segmento, sumando despues numeradores y
    denominadores de todos los segmentos sin crear pares artificiales entre
    ellos."""
    n_total = len(values)
    mean = float(values.mean()) if n_total else np.nan
    centered_c0 = float(np.sum((values - mean) ** 2)) if n_total else np.nan

    blocks = [b - mean for b in _segment_blocks(values, segments)]

    acf_vals = np.full(max_lag, np.nan)
    pairs = np.zeros(max_lag, dtype=np.int64)
    for k in range(1, max_lag + 1):
        numerator = 0.0
        n_pairs = 0
        for block in blocks:
            m = len(block)
            if m > k:
                numerator += float(np.dot(block[:-k], block[k:]))
                n_pairs += m - k
        acf_vals[k - 1] = numerator / centered_c0 if centered_c0 else np.nan
        pairs[k - 1] = n_pairs

    return {"n_total": n_total, "acf": acf_vals, "pairs": pairs}


def _ljung_box_gap_aware(acf_result: dict[str, Any], lags: int) -> tuple[float, float, int]:
    """Estadistico de Ljung-Box adaptado a una serie con limites de segmento
    legitimos (gaps reales entre jornadas/segmentos parciales). Reemplaza la
    formula clasica `Q = n(n+2) * sum r_k^2/(n-k)` -- valida solo para una
    serie unica sin interrupciones -- por
    `Q = n(n+2) * sum r_k^2 / pares_efectivos_k`, donde `pares_efectivos_k` es
    el numero real de pares (x_t, x_{t+k}) formados dentro de un mismo
    segmento (nunca `n-k`, que asume falsamente continuidad total). El
    prefactor `n(n+2)` usa el tamano total de muestra, igual que la formula
    clasica; el termino de varianza por lag usa el conteo real de pares, no
    una cuenta inflada."""
    n_total = acf_result["n_total"]
    r = acf_result["acf"][:lags]
    pairs = acf_result["pairs"][:lags]
    terms = np.where(pairs > 0, (r ** 2) / np.maximum(pairs, 1), 0.0)
    q_stat = n_total * (n_total + 2) * float(np.sum(terms))
    p_value = float(1 - _sps.chi2.cdf(q_stat, df=lags))
    return q_stat, p_value, int(pairs[-1]) if len(pairs) else 0


def build_acf_summary(df_ohlcv: pd.DataFrame, populations: list[str], config: dict[str, Any]) -> pd.DataFrame:
    series_names = config["dependence"]["series"]
    max_lag = config["dependence"]["max_lag"]
    min_obs_global = config["dependence"]["min_obs_global"]
    min_obs_regime = config["dependence"]["min_obs_regime"]

    rows: list[dict[str, Any]] = []
    for population in populations:
        df_pop = filter_population(df_ohlcv, population)
        _acf_rows_for_scope(df_pop, "global", None, None, series_names, max_lag, min_obs_global, rows, population)
        if "regime_id" in df_pop.columns:
            for rid, sub in df_pop.groupby("regime_id"):
                label = sub["regime_label"].iloc[0] if "regime_label" in sub.columns and len(sub) else str(rid)
                _acf_rows_for_scope(sub, "regime", int(rid), label, series_names, max_lag, min_obs_regime, rows, population)
    return pd.DataFrame(rows)


def _acf_rows_for_scope(sub, scope, regime_id, regime_label, series_names, max_lag, min_obs, rows, population) -> None:
    for series_name in series_names:
        if series_name not in sub.columns:
            continue
        values, segments = _clean_series_with_segment(sub, series_name)
        n_obs = len(values)
        if n_obs < min_obs:
            continue
        result = _gap_aware_acf(values, segments, max_lag)
        conf_bound = 1.96 / np.sqrt(n_obs)
        for lag in range(1, max_lag + 1):
            rows.append({
                "population": population, "scope": scope, "regime_id": regime_id, "regime_label": regime_label,
                "series": series_name, "lag": lag, "acf_value": float(result["acf"][lag - 1]),
                "conf_bound_95": float(conf_bound), "n_obs": n_obs,
                "n_pairs_effective": int(result["pairs"][lag - 1]),
            })


def build_dependence_tests_summary(df_ohlcv: pd.DataFrame, populations: list[str], config: dict[str, Any]) -> pd.DataFrame:
    series_names = config["dependence"]["series"]
    lb_lags = config["dependence"]["ljung_box_lags"]
    arch_lags = config["dependence"]["arch_lm_lags"]
    min_obs_global = config["dependence"]["min_obs_global"]
    min_obs_regime = config["dependence"]["min_obs_regime"]

    rows: list[dict[str, Any]] = []
    for population in populations:
        df_pop = filter_population(df_ohlcv, population)
        rows += _dependence_rows_for_scope(df_pop, "global", None, None, series_names, lb_lags, arch_lags, min_obs_global, population)
        if "regime_id" in df_pop.columns:
            for rid, sub in df_pop.groupby("regime_id"):
                label = sub["regime_label"].iloc[0] if "regime_label" in sub.columns and len(sub) else str(rid)
                rows += _dependence_rows_for_scope(sub, "regime", int(rid), label, series_names, lb_lags, arch_lags, min_obs_regime, population)
    return pd.DataFrame(rows)


def _dependence_rows_for_scope(sub, scope, regime_id, regime_label, series_names, lb_lags, arch_lags, min_obs, population) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    max_lb_lag = int(lb_lags[-1])
    for series_name in series_names:
        if series_name not in sub.columns:
            continue
        values, segments = _clean_series_with_segment(sub, series_name)
        n_obs = len(values)
        sufficient = n_obs >= min_obs

        lb_stat = lb_p = arch_stat = arch_p = np.nan
        n_pairs_max_lag = 0
        if sufficient:
            acf_result = _gap_aware_acf(values, segments, max_lb_lag)
            lb_stat, lb_p, n_pairs_max_lag = _ljung_box_gap_aware(acf_result, lags=max_lb_lag)
            # ARCH-LM (het_arch) NO es gap-aware: statsmodels arma la matriz
            # de regresores lageados sobre el array completo sin conocer
            # limites de segmento, por lo que hereda la misma contaminacion
            # entre segmentos que Ljung-Box tenia antes de esta correccion.
            # No se corrige aqui (fuera del alcance solicitado); queda
            # documentado como limitacion pendiente equivalente.
            try:
                arch_stat_, arch_p_, _, _ = het_arch(values, nlags=arch_lags)
                arch_stat, arch_p = float(arch_stat_), float(arch_p_)
            except (ValueError, np.linalg.LinAlgError):
                arch_stat = arch_p = np.nan

        out_rows.append({
            "population": population, "scope": scope, "regime_id": regime_id, "regime_label": regime_label,
            "series": series_name, "test_name": "ljung_box", "statistic": lb_stat, "p_value": lb_p,
            "lags_used": max_lb_lag, "n_obs": n_obs, "sufficient_sample": sufficient,
            "n_pairs_effective_at_max_lag": n_pairs_max_lag,
        })
        out_rows.append({
            "population": population, "scope": scope, "regime_id": regime_id, "regime_label": regime_label,
            "series": series_name, "test_name": "arch_lm", "statistic": arch_stat, "p_value": arch_p,
            "lags_used": int(arch_lags), "n_obs": n_obs, "sufficient_sample": sufficient,
        })
    return out_rows


# ---------------------------------------------------------------------------
# 10) Gobernanza: manifest, staleness, guardado atomico
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    config: dict[str, Any],
    s01_manifest: dict[str, Any],
    intraday_path: Path,
    audit_path: Path,
    validation: dict[str, Any],
    population_row_counts: dict[str, int],
    output_files: dict[str, dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    provenance = get_git_provenance(repo_root)
    return {
        "pipeline_version": config["pipeline_version"],
        "staleness": {
            "intraday_parquet_sha256": sha256_file(intraday_path),
            "trading_day_audit_sha256": sha256_file(audit_path),
            "s01_module_sha256": s01_manifest.get("staleness", {}).get("module_sha256"),
            "module_sha256": get_module_hash(),
            "config_sha256_normalized": sha256_bytes(normalized_config_bytes(config)),
            "pipeline_version": config["pipeline_version"],
            "force_rebuild": bool(config.get("force_rebuild", False)),
        },
        "provenance_metadata_only": {
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
        },
        "populations": config["populations"],
        "quantitative_populations": config["quantitative_populations"],
        "horizons_minutes": config["horizons_minutes"],
        "validation": validation,
        "population_row_counts": population_row_counts,
        "output_files": output_files,
    }


def staleness_fields_match(old: dict[str, Any] | None, new_manifest: dict[str, Any]) -> bool:
    if old is None:
        return False
    old_st = old.get("staleness", {})
    new_st = new_manifest["staleness"]
    if new_st["force_rebuild"]:
        return False
    keys = ["intraday_parquet_sha256", "trading_day_audit_sha256", "module_sha256", "config_sha256_normalized", "pipeline_version"]
    return all(old_st.get(k) == new_st.get(k) for k in keys)


def save_artifact(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    return atomic_write_parquet(df.reset_index(drop=True) if df.index.name is None else df, path)


# ---------------------------------------------------------------------------
# 11) Orquestacion
# ---------------------------------------------------------------------------

@dataclass
class S02Result:
    manifest: dict[str, Any]
    validation: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    df_ohlcv_by_population: dict[str, pd.DataFrame] = field(default_factory=dict)
    output_dir: Path | None = None
    manifest_path: Path | None = None
    reused_existing: bool = False


def run_s02_analysis(
    project_root: Path,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    force_rebuild: bool | None = None,
) -> S02Result:
    project_root = Path(project_root).resolve()
    config_path = Path(config_path) if config_path else project_root / "config" / "s02_analysis_config.yaml"
    config = load_config(config_path)
    if force_rebuild is not None:
        config["force_rebuild"] = force_rebuild

    s01 = load_s01_artifacts(project_root, config)
    df_raw = s01["df_intraday"]

    validation = validate_intraday_dataset(df_raw)
    if not validation["ok"]:
        raise IngestionError(
            f"Validacion temporal fallida. Checks criticos con error: {validation['failed_critical_checks']}"
        )

    df_tagged = attach_population_tags(df_raw, s01["df_audit"])
    df_ohlcv_all = build_ohlcv_metrics(df_tagged)

    populations = config["populations"]
    quantitative_populations = config["quantitative_populations"]
    horizons = config["horizons_minutes"]

    df_by_population = {pop: filter_population(df_ohlcv_all, pop) for pop in populations}
    population_row_counts = {pop: int(len(df)) for pop, df in df_by_population.items()}

    tables: dict[str, pd.DataFrame] = {}

    tables["s02_summary"] = pd.DataFrame([
        {
            "population": pop, "n_rows": len(df), "n_dates": int(df["date"].nunique()) if len(df) else 0,
            "first_timestamp": str(df.index.min()) if len(df) else None,
            "last_timestamp": str(df.index.max()) if len(df) else None,
            "avg_bars_per_day": (len(df) / df["date"].nunique()) if len(df) and df["date"].nunique() else np.nan,
            "pct_of_total_rows": len(df) / len(df_ohlcv_all) * 100.0 if len(df_ohlcv_all) else np.nan,
        }
        for pop, df in df_by_population.items()
    ])

    tables["s02_coverage_summary"] = build_coverage_summary(df_ohlcv_all)
    tables["s02_regime_distribution"] = build_regime_distribution_summary(
        {pop: df_by_population[pop] for pop in quantitative_populations}
    )
    tables["s02_ohlcv_stats_summary"] = build_ohlcv_stats_summary(df_by_population)
    tables["s02_ohlcv_correlation"] = build_ohlcv_correlation(df_by_population[POPULATION_FULL_DAY], POPULATION_FULL_DAY)

    df_valid_by_population: dict[str, pd.DataFrame] = {}
    window_validity_frames = []
    rollover_frames = []
    window_metrics_summary_frames = []
    stability_frames = []
    rolling_frames = []
    instability_frames = []

    for population in quantitative_populations:
        df_pop = df_by_population[population]
        df_valid = build_window_validity(df_pop, horizons)
        df_valid = build_historical_window_metrics(df_valid, horizons)
        df_valid = build_future_window_metrics(df_valid, horizons)
        df_valid_by_population[population] = df_valid

        window_validity_frames.append(build_window_validity_summary(df_valid, horizons, population))
        rollover_audit = build_rollover_window_audit(df_valid, horizons)
        rollover_audit.insert(0, "population", population)
        rollover_frames.append(rollover_audit)

        wms = build_window_metrics_summary(df_valid, horizons, population)
        window_metrics_summary_frames.append(wms)

        stability_frames.append(build_stability_summary(wms, population))

        daily = build_daily_aggregates(df_valid, horizons)
        rolling = build_rolling_summary(daily, horizons, config["rolling_windows_days"], config["rolling_min_period_ratio"])
        rolling.insert(0, "population", population)
        rolling_frames.append(rolling)

        instability = build_temporal_instability_zones(
            daily, horizons, config["rolling_windows_days"], config["rolling_min_period_ratio"],
            config["temporal_instability_zone_percentiles"],
        )
        instability.insert(0, "population", population)
        instability_frames.append(instability)

    tables["s02_window_validity_summary"] = pd.concat(window_validity_frames, ignore_index=True)
    tables["s02_rollover_window_audit"] = pd.concat(rollover_frames, ignore_index=True)
    tables["s02_window_metrics_summary"] = pd.concat(window_metrics_summary_frames, ignore_index=True)
    tables["s02_stability_summary"] = pd.concat(stability_frames, ignore_index=True)
    tables["s02_rolling_summary"] = pd.concat(rolling_frames, ignore_index=True)
    tables["s02_temporal_instability_zones"] = pd.concat(instability_frames, ignore_index=True)

    dependence_populations = [POPULATION_FULL_DAY]
    tables["s02_acf_summary"] = build_acf_summary(df_ohlcv_all, dependence_populations, config)
    tables["s02_dependence_tests_summary"] = build_dependence_tests_summary(df_ohlcv_all, dependence_populations, config)

    artifacts_cfg = config["artifacts"]
    out_dir = Path(output_dir) if output_dir else project_root / artifacts_cfg["intraday_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {
        "s02_summary": artifacts_cfg["summary_name"],
        "s02_coverage_summary": artifacts_cfg["coverage_name"],
        "s02_regime_distribution": artifacts_cfg["regime_distribution_name"],
        "s02_ohlcv_stats_summary": artifacts_cfg["ohlcv_stats_name"],
        "s02_ohlcv_correlation": artifacts_cfg["ohlcv_correlation_name"],
        "s02_window_validity_summary": artifacts_cfg["window_validity_name"],
        "s02_rollover_window_audit": artifacts_cfg["rollover_audit_name"],
        "s02_window_metrics_summary": artifacts_cfg["window_metrics_summary_name"],
        "s02_stability_summary": artifacts_cfg["stability_summary_name"],
        "s02_rolling_summary": artifacts_cfg["rolling_summary_name"],
        "s02_temporal_instability_zones": artifacts_cfg["temporal_instability_zones_name"],
        "s02_acf_summary": artifacts_cfg["acf_summary_name"],
        "s02_dependence_tests_summary": artifacts_cfg["dependence_tests_name"],
    }

    output_files: dict[str, dict[str, Any]] = {}
    for key, filename in name_map.items():
        path = out_dir / filename
        write_result = save_artifact(tables[key], path)
        output_files[key] = {"path": str(path), "rows": int(len(tables[key])), "sha256": write_result["sha256"]}

    manifest_path = out_dir / artifacts_cfg["manifest_name"]
    manifest = build_manifest(
        config=config, s01_manifest=s01["s01_manifest"], intraday_path=s01["intraday_path"],
        audit_path=s01["audit_path"], validation=validation, population_row_counts=population_row_counts,
        output_files=output_files, repo_root=project_root,
    )
    write_json_atomic(manifest, manifest_path)

    return S02Result(
        manifest=manifest, validation=validation, tables=tables,
        df_ohlcv_by_population=df_valid_by_population | {
            pop: df_by_population[pop] for pop in populations if pop not in df_valid_by_population
        },
        output_dir=out_dir, manifest_path=manifest_path, reused_existing=False,
    )
