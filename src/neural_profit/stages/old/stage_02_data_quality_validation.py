# stage_02_data_quality_validation.py
# ------------------------------------------------------------
# Stage_02: Data quality gate para dataset intradía MNQ.
#
# Objetivo:
# - Validar que el dataset intradía procesado (stage_01) cumpla criterios mínimos
#   de calidad antes de continuar con stages de targets/features/modelos.
#
# Puntos clave (temporalidad):
# - Este stage NO genera features ni targets.
# - Se permite "bajar" el DatetimeIndex a columna con reset_index SOLO para validar,
#   pero se preserva trazabilidad temporal ordenando explícitamente por (date, datetime)
#   y registrando checks de monotonía / duplicados.
#
# Output:
# - JSON con PASS/FAIL + métricas + checks + ejemplos de filas inválidas.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import logging


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("stage_02")


# ------------------------------------------------------------
# MLflow (opcional)
# ------------------------------------------------------------
try:
    import mlflow
except Exception:
    mlflow = None


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _nan_ratio(s: pd.Series) -> float:
    """Ratio de NaNs en una serie."""
    return float(s.isna().mean())


def _write_report(output_path: str | Path, report: Dict[str, Any]) -> None:
    """Escribe el reporte JSON en disco (siempre)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame tenga una columna 'datetime' usable.

    Reglas:
    - Si el índice es DatetimeIndex: lo baja a columna con reset_index.
      * Si ya existía una columna 'datetime' (colisión), se prioriza el índice
        (porque es la fuente temporal principal del pipeline).
    - Si NO hay DatetimeIndex y NO hay columna 'datetime': se considera formato inválido.
    """
    df = df.copy()

    # Caso 1: DatetimeIndex => bajamos a columna (para validación)
    if isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name or "index"

        # Blindaje contra colisión: si ya existe 'datetime' como columna
        if "datetime" in df.columns and idx_name != "datetime":
            # Bajamos índice a columna auxiliar
            df = df.reset_index().rename(columns={idx_name: "datetime_index"})
            # Priorizamos el índice (fuente temporal) como datetime definitivo
            df["datetime"] = df["datetime_index"]
            df.drop(columns=["datetime_index"], inplace=True)
        else:
            # Caso típico: el índice se llama 'datetime' o no hay 'datetime' en columnas
            df = df.reset_index().rename(columns={idx_name: "datetime"})

    # Caso 2: No DatetimeIndex => exigimos columna datetime
    if "datetime" not in df.columns:
        raise TypeError("No 'datetime' column found (expected it as DatetimeIndex or column).")

    # Parse robusto por si datetime viene como string/object
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


def _log_temporal_audit(df: pd.DataFrame, label: str) -> None:
    """
    Logs de auditoría temporal (NO modifican datos).
    Útiles para detectar problemas de:
    - duplicados
    - orden no-monótono por día
    """
    # Duplicados globales en datetime
    n_dup_global = int(df["datetime"].duplicated().sum())
    log.info(f"[CHECK:{label}] datetime duplicates (global): {n_dup_global}")

    # Monotonía por día (tras ordenar, debería ser True para todos)
    if "date" in df.columns:
        non_monotonic_days = (
            df.groupby("date")["datetime"]
            .apply(lambda s: not s.is_monotonic_increasing)
        )
        n_bad_days = int(non_monotonic_days.sum()) if len(non_monotonic_days) else 0
        log.info(f"[CHECK:{label}] days with non-monotonic datetime: {n_bad_days}")

        if n_bad_days > 0:
            sample_days = non_monotonic_days[non_monotonic_days].index[:10].tolist()
            log.warning(f"[WARN:{label}] Ejemplos de días no-monótonos: {sample_days}")


# ------------------------------------------------------------
# Core validation
# ------------------------------------------------------------
def validate_intraday_quality(
    input_path: str,
    output_path: str,
    expected_freq_seconds: int = 60,
    expected_minutes_per_day: int = 451,
    max_incomplete_days_ratio: float = 0.01,
    max_gap_seconds: int = 60,
    max_total_nan_ratio: float = 0.001,
    max_col_nan_ratio: float = 0.002,
) -> Dict[str, Any]:
    """
    Valida calidad de un dataset intradía y produce un reporte JSON (PASS/FAIL).

    Suposiciones del dataset:
    - Timestamp puede venir como DatetimeIndex (tz-aware o tz-naive) o como columna 'datetime'.
    - Columnas mínimas: open, high, low, close, volume.

    NOTA sobre reset_index:
    - Se utiliza únicamente para convertir el índice temporal en columna y poder validar.
    - No se altera el contenido temporal: se ordena explícitamente por (date, datetime).
    """

    # -----------------------------
    # 1) Leer dataset
    # -----------------------------
    if not Path(input_path).exists():
        report = {
            "pass": False,
            "reason": f"Input parquet not found: {input_path}",
            "metrics": {},
            "checks": {},
            "invalid_examples": [],
        }
        _write_report(output_path, report)
        return report

    df = pd.read_parquet(input_path).copy()

    # -----------------------------
    # 2) Asegurar columna datetime (con blindaje)
    # -----------------------------
    try:
        df = _ensure_datetime_column(df)
    except Exception as e:
        report = {
            "pass": False,
            "reason": f"Failed to ensure datetime column: {type(e).__name__}: {e}",
            "metrics": {},
            "checks": {"datetime_column_ok": False},
            "invalid_examples": [],
        }
        _write_report(output_path, report)
        return report

    # Fail temprano: datetimes inválidos (NaT)
    if df["datetime"].isna().any():
        bad_count = int(df["datetime"].isna().sum())
        report = {
            "pass": False,
            "reason": f"Found {bad_count} invalid datetime values (NaT).",
            "metrics": {"invalid_datetime_count": bad_count},
            "checks": {"datetime_parse_ok": False},
            "invalid_examples": [],
        }
        _write_report(output_path, report)
        return report

    # -----------------------------
    # 3) Crear columna date (día de trading)
    # -----------------------------
    df["date"] = df["datetime"].dt.date

    # -----------------------------
    # 4) Validar columnas OHLCV mínimas
    # -----------------------------
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        report = {
            "pass": False,
            "reason": f"Missing required columns: {missing}",
            "metrics": {},
            "checks": {"required_cols_ok": False},
            "invalid_examples": [],
        }
        _write_report(output_path, report)
        return report

    # -----------------------------
    # 5) Orden explícito (preserva trazabilidad temporal)
    # -----------------------------
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    # Auditoría temporal (logs)
    _log_temporal_audit(df, label="post_sort")

    # -----------------------------
    # 6) Métricas NaNs
    # -----------------------------
    # Nota: incluimos todas las columnas actuales del df, incluyendo 'date'
    # y futuras columnas auxiliares. Si prefiere, puede filtrar solo OHLCV.
    col_nan = {c: _nan_ratio(df[c]) for c in df.columns}
    total_nan_ratio = float(df.isna().mean().mean())

    # -----------------------------
    # 7) Duplicados por (date, datetime)
    # -----------------------------
    dup_mask = df.duplicated(subset=["date", "datetime"], keep=False)
    dup_count = int(dup_mask.sum())

    # -----------------------------
    # 8) Días incompletos (por conteo de registros/día)
    # -----------------------------
    counts_per_day = df.groupby("date")["datetime"].count()
    days_total = int(counts_per_day.shape[0])

    incomplete_days = counts_per_day[counts_per_day < expected_minutes_per_day]
    incomplete_days_count = int(incomplete_days.shape[0])
    incomplete_days_ratio = float(incomplete_days_count / days_total) if days_total else 1.0

    # -----------------------------
    # 9) Gaps intradía
    #    diff entre timestamps consecutivos dentro del mismo día (segundos)
    # -----------------------------
    df["ts_diff_sec"] = df.groupby("date")["datetime"].diff().dt.total_seconds()

    # gaps mayores a expected_freq_seconds para estimar el máximo encontrado
    gap_mask = df["ts_diff_sec"] > expected_freq_seconds

    # gaps mayores al máximo permitido => FAIL si hay alguno
    gaps_over_max = df["ts_diff_sec"] > max_gap_seconds
    gaps_count = int(gaps_over_max.sum())
    max_gap_found = float(df.loc[gap_mask, "ts_diff_sec"].max()) if gap_mask.any() else 0.0

    # -----------------------------
    # 10) Valores inválidos OHLCV
    # -----------------------------
    invalid_mask = (
        (df["high"] < df["low"]) |
        (df["close"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["volume"] < 0)
    )
    invalid_count = int(invalid_mask.sum())

    invalid_examples: List[Dict[str, Any]] = []
    if invalid_count > 0:
        sample = df.loc[
            invalid_mask,
            ["date", "datetime", "open", "high", "low", "close", "volume"]
        ].head(20)
        invalid_examples = sample.astype(str).to_dict(orient="records")

    # -----------------------------
    # 11) Checks + PASS/FAIL
    # -----------------------------
    checks = {
        "datetime_parse_ok": True,
        "required_cols_ok": True,
        "total_nan_ratio_ok": total_nan_ratio <= max_total_nan_ratio,
        "col_nan_ratio_ok": all(v <= max_col_nan_ratio for v in col_nan.values()),
        "incomplete_days_ratio_ok": incomplete_days_ratio <= max_incomplete_days_ratio,
        "gaps_ok": gaps_count == 0,
        "duplicates_ok": dup_count == 0,
        "invalid_values_ok": invalid_count == 0,
    }
    passed = all(checks.values())

    report: Dict[str, Any] = {
        "pass": bool(passed),
        "reason": None if passed else "Quality checks failed (see checks/metrics).",
        "metrics": {
            "days_total": days_total,
            "incomplete_days_count": incomplete_days_count,
            "incomplete_days_ratio": incomplete_days_ratio,
            "gaps_count": gaps_count,
            "max_gap_seconds_found": max_gap_found,
            "duplicates_count": dup_count,
            "invalid_values_count": invalid_count,
            "total_nan_ratio": total_nan_ratio,
            **{f"nan_ratio__{k}": float(v) for k, v in col_nan.items()},
        },
        "checks": checks,
        "invalid_examples": invalid_examples,
    }

    # -----------------------------
    # 12) Guardar JSON
    # -----------------------------
    _write_report(output_path, report)

    return report


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI)")
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input parquet (ej: data/processed/mnq_intraday.parquet)")
    p.add_argument("--output", required=True, help="Output report json (ej: reports/data_validation.json)")

    p.add_argument("--expected_freq_seconds", type=int, default=60)
    p.add_argument("--expected_minutes_per_day", type=int, default=451)
    p.add_argument("--max_incomplete_days_ratio", type=float, default=0.01)
    p.add_argument("--max_gap_seconds", type=int, default=60)
    p.add_argument("--max_total_nan_ratio", type=float, default=0.001)
    p.add_argument("--max_col_nan_ratio", type=float, default=0.002)

    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("[0.1] Output report path: %s", out)

    log.info("[1] Ejecutando validación de calidad (quality gate)")
    log.info(
        "[1.1] Params: freq=%ss, minutes/day=%s, max_incomplete_ratio=%.4f, max_gap=%ss, "
        "max_total_nan=%.6f, max_col_nan=%.6f",
        args.expected_freq_seconds,
        args.expected_minutes_per_day,
        args.max_incomplete_days_ratio,
        args.max_gap_seconds,
        args.max_total_nan_ratio,
        args.max_col_nan_ratio,
    )

    try:
        report = validate_intraday_quality(
            input_path=args.input,
            output_path=args.output,
            expected_freq_seconds=args.expected_freq_seconds,
            expected_minutes_per_day=args.expected_minutes_per_day,
            max_incomplete_days_ratio=args.max_incomplete_days_ratio,
            max_gap_seconds=args.max_gap_seconds,
            max_total_nan_ratio=args.max_total_nan_ratio,
            max_col_nan_ratio=args.max_col_nan_ratio,
        )

        passed = bool(report.get("pass"))
        reason = report.get("reason", "")
        log.info("[2] Validación finalizada: pass=%s", passed)
        if reason:
            log.info("[2.1] Reason: %s", reason)

        # MLflow logging (opcional)
        if mlflow is not None:
            log.info("[3] MLflow disponible: logueando métricas/tags")
            with mlflow.start_run(run_name="stage_02_data_quality_validation"):
                for k, v in report.get("metrics", {}).items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        mlflow.log_metric(k, float(v))
                mlflow.set_tag("quality_pass", str(passed))
                for ck, ok in report.get("checks", {}).items():
                    mlflow.set_tag(f"check__{ck}", str(ok))
        else:
            log.info("[3] MLflow no disponible: omitido")

    except Exception as e:
        log.exception("[ERR] Excepción durante validate_intraday_quality")
        report = {
            "pass": False,
            "reason": f"Exception: {type(e).__name__}: {e}",
            "metrics": {},
            "checks": {},
            "invalid_examples": [],
        }
        _write_report(args.output, report)
        log.info("[4] Report escrito pese a excepción: %s", out)

    # Quality gate real: FAIL => exit code 1
    passed = bool(report.get("pass"))
    log.info("[5] Exit code = %s", 0 if passed else 1)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
