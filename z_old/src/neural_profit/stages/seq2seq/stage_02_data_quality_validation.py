"""
stage_02_data_quality_validation.py

Contrato del Stage
------------------
Propósito:
- Ejecutar un "quality gate" sobre el dataset intradía procesado (stage_01)
  antes de continuar con stages de targets/features/modelos.

Inputs (deps DVC):
- data/processed/mnq_intraday.parquet

Outputs (outs DVC):
- (no genera dataset) -> solo reportes/metrics

Reports (auditoría):
- reports/stage_02_data_quality_report.json   (PASS/FAIL + métricas + checks + ejemplos)

Params:
- EXPECTED_FREQ_SECONDS
- EXPECTED_MINUTES_PER_DAY
- MAX_INCOMPLETE_DAYS_RATIO
- MAX_GAP_SECONDS
- MAX_TOTAL_NAN_RATIO
- MAX_COL_NAN_RATIO
- ENABLE_MLFLOW

Notas:
- Este stage NO crea features ni targets.
- Se permite bajar DatetimeIndex a columna 'datetime' SOLO para validar;
  se preserva trazabilidad ordenando explícitamente por (date, datetime).
- Si PASS=False, el proceso termina con exit code 1 (corta el pipeline).
- MLflow es opcional y, si está habilitado, loguea params/metrics + JSON como artifact.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Imports (stdlib)
# ---------------------------------------------------------------------
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

# ---------------------------------------------------------------------
# Imports (third-party)
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging (uniforme)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_02")

# ---------------------------------------------------------------------
# Configuración de rutas (DVC-friendly) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
IN_INTRADAY_PARQUET = Path(
    os.environ.get("IN_INTRADAY_PARQUET", "data/processed/mnq_intraday.parquet")
)
REPORT_QUALITY = Path(
    os.environ.get("REPORT_QUALITY", "reports/stage_02_data_quality_report.json")
)

# ---------------------------------------------------------------------
# Configuración funcional (params reproducibles) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
STAGE_NAME = "stage_02_data_quality_validation"
STAGE_VERSION = "1.0"

EXPECTED_FREQ_SECONDS = int(os.environ.get("EXPECTED_FREQ_SECONDS", "60"))
EXPECTED_MINUTES_PER_DAY = int(os.environ.get("EXPECTED_MINUTES_PER_DAY", "451"))
MAX_INCOMPLETE_DAYS_RATIO = float(os.environ.get("MAX_INCOMPLETE_DAYS_RATIO", "0.01"))
MAX_GAP_SECONDS = int(os.environ.get("MAX_GAP_SECONDS", "60"))
MAX_TOTAL_NAN_RATIO = float(os.environ.get("MAX_TOTAL_NAN_RATIO", "0.001"))
MAX_COL_NAN_RATIO = float(os.environ.get("MAX_COL_NAN_RATIO", "0.002"))

ENABLE_MLFLOW = os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "yes", "YES"}

# Columnas mínimas esperadas (contractual)
REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------
# Utilidades generales (reusables)
# ---------------------------------------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    _ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_summary_envelope(
    *,
    stage: str,
    version: str,
    inputs: Dict[str, str],
    outputs: Dict[str, str],
    reports: Dict[str, str],
    params: Dict[str, Any],
    metrics: Dict[str, float],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "created_at_utc": _utc_now_iso(),
        "version": version,
        "paths": {"inputs": inputs, "outputs": outputs, "reports": reports},
        "params": params or {},
        "metrics": metrics or {},
        "details": details or {},
    }


def print_summary_console(summary: Dict[str, Any]) -> None:
    def _fmt(x: Any) -> str:
        return "N/A" if x is None else str(x)

    print("\n" + "=" * 70)
    print(f"STAGE: {_fmt(summary.get('stage'))}")
    print(f"CREATED_AT_UTC: {_fmt(summary.get('created_at_utc'))}")
    print(f"VERSION: {_fmt(summary.get('version'))}")
    print("-" * 70)

    print("[PARAMS]")
    for k, v in (summary.get("params", {}) or {}).items():
        print(f"  - {k}: {v}")

    print("[METRICS]")
    for k, v in (summary.get("metrics", {}) or {}).items():
        print(f"  - {k}: {v}")

    print("=" * 70 + "\n")


def mlflow_log_from_summary(
    *,
    enable: bool,
    stage: str,
    summary: Dict[str, Any],
    summary_path: Path,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    MLflow opcional.
    Loguea params/metrics + adjunta JSON summary como artifact.
    """
    if not enable:
        return

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise ImportError("MLflow no está instalado. Instale con: pip install mlflow") from exc

    if run_name is None:
        run_name = stage

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", stage)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        for k, v in (summary.get("params", {}) or {}).items():
            mlflow.log_param(k, v)

        for k, v in (summary.get("metrics", {}) or {}).items():
            if isinstance(v, (int, float)) and v == v and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        mlflow.log_artifact(str(summary_path))


# ---------------------------------------------------------------------
# Helpers de validación (puras)
# ---------------------------------------------------------------------
def _nan_ratio(s: pd.Series) -> float:
    return float(s.isna().mean())


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columna 'datetime' usable para validación.
    - Si índice es DatetimeIndex: reset_index.
    - Si ya existe 'datetime' como columna (colisión): prioriza el índice.
    - Si no hay DatetimeIndex ni columna datetime: FAIL.
    """
    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        idx_name = df.index.name or "index"

        if "datetime" in df.columns and idx_name != "datetime":
            df = df.reset_index().rename(columns={idx_name: "datetime_index"})
            df["datetime"] = df["datetime_index"]
            df.drop(columns=["datetime_index"], inplace=True)
        else:
            df = df.reset_index().rename(columns={idx_name: "datetime"})

    if "datetime" not in df.columns:
        raise TypeError("No 'datetime' column found (expected it as DatetimeIndex or column).")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _log_temporal_audit(df: pd.DataFrame, label: str) -> Tuple[int, int, List]:
    """
    Logs de auditoría temporal (NO modifican datos).
    Retorna:
      - n_dup_global: duplicados globales en datetime
      - n_bad_days: días con no-monotonía en datetime
      - sample_days: ejemplos (hasta 10)
    """
    n_dup_global = int(df["datetime"].duplicated().sum())
    log.info("[CHECK:%s] datetime duplicates (global): %s", label, n_dup_global)

    n_bad_days = 0
    sample_days: List = []
    if "date" in df.columns:
        non_monotonic_days = df.groupby("date")["datetime"].apply(lambda s: not s.is_monotonic_increasing)
        n_bad_days = int(non_monotonic_days.sum()) if len(non_monotonic_days) else 0
        log.info("[CHECK:%s] days with non-monotonic datetime: %s", label, n_bad_days)

        if n_bad_days > 0:
            sample_days = non_monotonic_days[non_monotonic_days].index[:10].tolist()
            log.warning("[WARN:%s] Ejemplos de días no-monótonos: %s", label, sample_days)

    return n_dup_global, n_bad_days, sample_days


# ---------------------------------------------------------------------
# Core validation (pura: NO escribe en disco)
# ---------------------------------------------------------------------
def validate_intraday_quality(
    *,
    df: pd.DataFrame,
    expected_freq_seconds: int,
    expected_minutes_per_day: int,
    max_incomplete_days_ratio: float,
    max_gap_seconds: int,
    max_total_nan_ratio: float,
    max_col_nan_ratio: float,
) -> Tuple[bool, str, Dict[str, float], Dict[str, Any]]:
    """
    Ejecuta las validaciones y devuelve:
      - passed (bool)
      - reason (str)
      - metrics (dict[str,float])  -> MLflow-friendly
      - details (dict)            -> JSON-serializable (checks + ejemplos)
    """
    # -----------------------------
    # 1) datetime + fail temprano
    # -----------------------------
    df = _ensure_datetime_column(df)

    invalid_dt = int(df["datetime"].isna().sum())
    if invalid_dt > 0:
        metrics = {"invalid_datetime_count": float(invalid_dt)}
        details = {
            "checks": {"datetime_parse_ok": False},
            "invalid_examples": [],
        }
        return False, f"Found {invalid_dt} invalid datetime values (NaT).", metrics, details

    # -----------------------------
    # 2) date + required cols
    # -----------------------------
    df["date"] = df["datetime"].dt.date

    missing = sorted(list(set(REQUIRED_OHLCV) - set(df.columns)))
    if missing:
        details = {"checks": {"required_cols_ok": False}, "missing_cols": missing, "invalid_examples": []}
        return False, f"Missing required columns: {missing}", {}, details

    # -----------------------------
    # 3) orden temporal explícito
    # -----------------------------
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    n_dup_global, n_bad_days, sample_bad_days = _log_temporal_audit(df, label="post_sort")

    # -----------------------------
    # 4) NaNs
    # -----------------------------
    col_nan = {c: _nan_ratio(df[c]) for c in df.columns}
    total_nan_ratio = float(df.isna().mean().mean())

    # -----------------------------
    # 5) duplicados por (date, datetime)
    # -----------------------------
    dup_mask = df.duplicated(subset=["date", "datetime"], keep=False)
    dup_count = int(dup_mask.sum())

    # -----------------------------
    # 6) días incompletos (conteo)
    # -----------------------------
    counts_per_day = df.groupby("date")["datetime"].count()
    days_total = int(counts_per_day.shape[0])

    incomplete_days = counts_per_day[counts_per_day < expected_minutes_per_day]
    incomplete_days_count = int(incomplete_days.shape[0])
    incomplete_days_ratio = float(incomplete_days_count / days_total) if days_total else 1.0

    # -----------------------------
    # 7) gaps intradía
    # -----------------------------
    df["ts_diff_sec"] = df.groupby("date")["datetime"].diff().dt.total_seconds()

    gaps_over_max = df["ts_diff_sec"] > max_gap_seconds
    gaps_count = int(gaps_over_max.sum())

    gap_mask = df["ts_diff_sec"] > expected_freq_seconds
    max_gap_found = float(df.loc[gap_mask, "ts_diff_sec"].max()) if gap_mask.any() else 0.0

    # -----------------------------
    # 8) valores inválidos OHLCV
    # -----------------------------
    invalid_mask = (
        (df["high"] < df["low"])
        | (df["close"] < df["low"])
        | (df["close"] > df["high"])
        | (df["volume"] < 0)
    )
    invalid_count = int(invalid_mask.sum())

    invalid_examples: List[Dict[str, Any]] = []
    if invalid_count > 0:
        sample = df.loc[
            invalid_mask,
            ["date", "datetime", "open", "high", "low", "close", "volume"],
        ].head(20)
        invalid_examples = sample.astype(str).to_dict(orient="records")

    # -----------------------------
    # 9) checks + PASS/FAIL
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
        # Auditoría extra (no gatea por defecto, pero queda registrado)
        "audit_days_non_monotonic_ok": n_bad_days == 0,
        "audit_global_datetime_duplicates_ok": n_dup_global == 0,
    }

    passed = all(
        checks[k]
        for k in [
            "datetime_parse_ok",
            "required_cols_ok",
            "total_nan_ratio_ok",
            "col_nan_ratio_ok",
            "incomplete_days_ratio_ok",
            "gaps_ok",
            "duplicates_ok",
            "invalid_values_ok",
        ]
    )

    reason = None if passed else "Quality checks failed (see checks/metrics)."

    # -----------------------------
    # 10) metrics (MLflow-friendly)
    # -----------------------------
    metrics: Dict[str, float] = {
        "pass": 1.0 if passed else 0.0,
        "days_total": float(days_total),
        "incomplete_days_count": float(incomplete_days_count),
        "incomplete_days_ratio": float(incomplete_days_ratio),
        "gaps_count": float(gaps_count),
        "max_gap_seconds_found": float(max_gap_found),
        "duplicates_count": float(dup_count),
        "invalid_values_count": float(invalid_count),
        "total_nan_ratio": float(total_nan_ratio),
        "audit_global_datetime_duplicates": float(n_dup_global),
        "audit_days_non_monotonic": float(n_bad_days),
    }
    # ratios por columna (prefijo estable)
    metrics.update({f"nan_ratio__{k}": float(v) for k, v in col_nan.items()})

    # -----------------------------
    # 11) details (JSON-serializable)
    # -----------------------------
    details: Dict[str, Any] = {
        "reason": reason,
        "checks": checks,
        "thresholds": {
            "expected_freq_seconds": expected_freq_seconds,
            "expected_minutes_per_day": expected_minutes_per_day,
            "max_incomplete_days_ratio": max_incomplete_days_ratio,
            "max_gap_seconds": max_gap_seconds,
            "max_total_nan_ratio": max_total_nan_ratio,
            "max_col_nan_ratio": max_col_nan_ratio,
        },
        "audit": {
            "datetime_duplicates_global": n_dup_global,
            "days_non_monotonic": n_bad_days,
            "sample_days_non_monotonic": sample_bad_days,
        },
        "invalid_examples": invalid_examples,
        "incomplete_days_sample": {
            "n": int(incomplete_days_count),
            "dates": [str(d) for d in incomplete_days.index[:20].tolist()] if incomplete_days_count else [],
        },
    }

    return bool(passed), (reason or ""), metrics, details


# ---------------------------------------------------------------------
# CLI (argparse) - override (no inventa defaults)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage_02: Intraday data quality validation (gate).")

    p.add_argument("--in-intraday-parquet", type=str, default=str(IN_INTRADAY_PARQUET))
    p.add_argument("--report-quality", type=str, default=str(REPORT_QUALITY))

    p.add_argument("--expected-freq-seconds", type=int, default=int(EXPECTED_FREQ_SECONDS))
    p.add_argument("--expected-minutes-per-day", type=int, default=int(EXPECTED_MINUTES_PER_DAY))
    p.add_argument("--max-incomplete-days-ratio", type=float, default=float(MAX_INCOMPLETE_DAYS_RATIO))
    p.add_argument("--max-gap-seconds", type=int, default=int(MAX_GAP_SECONDS))
    p.add_argument("--max-total-nan-ratio", type=float, default=float(MAX_TOTAL_NAN_RATIO))
    p.add_argument("--max-col-nan-ratio", type=float, default=float(MAX_COL_NAN_RATIO))

    p.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=ENABLE_MLFLOW,
        help="Si se activa, registra params/métricas en MLflow",
    )

    # Aliases (compatibilidad con versiones viejas)
    p.add_argument("--input", type=str, dest="in_intraday_parquet", help=argparse.SUPPRESS)
    p.add_argument("--output", type=str, dest="report_quality", help=argparse.SUPPRESS)

    return p.parse_args()


# ---------------------------------------------------------------------
# Main (orquestación)
# ---------------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_parquet = Path(args.in_intraday_parquet)
    report_path = Path(args.report_quality)

    params: Dict[str, Any] = {
        "in_intraday_parquet": str(in_parquet.as_posix()),
        "report_quality": str(report_path.as_posix()),
        "expected_freq_seconds": int(args.expected_freq_seconds),
        "expected_minutes_per_day": int(args.expected_minutes_per_day),
        "max_incomplete_days_ratio": float(args.max_incomplete_days_ratio),
        "max_gap_seconds": int(args.max_gap_seconds),
        "max_total_nan_ratio": float(args.max_total_nan_ratio),
        "max_col_nan_ratio": float(args.max_col_nan_ratio),
    }

    log.info("[1] Cargando parquet intradía: %s", in_parquet)
    if not in_parquet.exists():
        # Report consistente + corte del pipeline
        summary = build_summary_envelope(
            stage=STAGE_NAME,
            version=STAGE_VERSION,
            inputs={"mnq_intraday_parquet": str(in_parquet.as_posix())},
            outputs={},
            reports={"quality_report": str(report_path.as_posix())},
            params=params,
            metrics={"pass": 0.0},
            details={"reason": f"Input parquet not found: {in_parquet.as_posix()}"},
        )
        save_json(summary, report_path)
        print_summary_console(summary)
        log.error("[FAIL] Input parquet not found. Exit code=1")
        sys.exit(1)

    df = pd.read_parquet(in_parquet).copy()

    log.info("[2] Ejecutando validación (quality gate)")
    passed, reason, metrics, details = validate_intraday_quality(
        df=df,
        expected_freq_seconds=int(args.expected_freq_seconds),
        expected_minutes_per_day=int(args.expected_minutes_per_day),
        max_incomplete_days_ratio=float(args.max_incomplete_days_ratio),
        max_gap_seconds=int(args.max_gap_seconds),
        max_total_nan_ratio=float(args.max_total_nan_ratio),
        max_col_nan_ratio=float(args.max_col_nan_ratio),
    )

    # Envelope estándar
    summary = build_summary_envelope(
        stage=STAGE_NAME,
        version=STAGE_VERSION,
        inputs={"mnq_intraday_parquet": str(in_parquet.as_posix())},
        outputs={},
        reports={"quality_report": str(report_path.as_posix())},
        params=params,
        metrics=metrics,
        details=details,
    )

    log.info("[3] Guardando report JSON: %s", report_path)
    save_json(summary, report_path)
    print_summary_console(summary)

    log.info("[4] MLflow tracking (enable=%s)", bool(args.enable_mlflow))
    mlflow_log_from_summary(
        enable=bool(args.enable_mlflow),
        stage=STAGE_NAME,
        summary=summary,
        summary_path=report_path,
        tags={"pipeline": "neural_profit", "dataset": "MNQ", "quality_gate": "true"},
    )

    # Quality gate real: FAIL => exit code 1
    log.info("[5] Quality gate: pass=%s", passed)
    if not passed:
        log.error("[FAIL] %s", reason or "Quality checks failed.")
        log.info("[5.1] Exit code=1")
        sys.exit(1)

    log.info("[OK] Stage_02 passed. Exit code=0")
    sys.exit(0)


# ---------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
