"""
stage_01_intraday_data_preparation.py

Contrato del Stage
------------------
Propósito:
- Preparar el dataset intradía de MNQ a partir del raw:
  (1) normaliza timezone, (2) filtra días hábiles NASDAQ,
  (3) filtra horario de trading, (4) elimina días incompletos / con gaps,
  y (5) guarda un parquet intradía consistente por día.

Inputs (deps DVC):
- data/raw/mnq_raw.parquet

Outputs (outs DVC):
- data/processed/mnq_intraday.parquet

Reports (auditoría):
- reports/stage_01_dataset_prep_summary.json

Params:
- MARKET, TRADING_START, TRADING_END, TZ_FROM, TZ_TO, GAP_MINUTES, ENABLE_MLFLOW
- Paths DVC-friendly: IN_RAW_PARQUET, OUT_PARQUET, REPORT_SUMMARY

Notas:
- El report JSON (summary envelope) es la fuente de verdad del stage.
- MLflow es opcional: si ENABLE_MLFLOW=1 o --enable-mlflow, loguea params/metrics y adjunta el JSON.
- Se asume que el raw contiene OHLCV (minuto a minuto) y un DatetimeIndex (o columna 'datetime').
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Imports (stdlib)
# ---------------------------------------------------------------------
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

# ---------------------------------------------------------------------
# Imports (third-party)
# ---------------------------------------------------------------------
import pandas as pd

# Import del calendario de mercado (requerido)
try:
    import pandas_market_calendars as mcal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: pandas_market_calendars. "
        "Install with: pip install pandas-market-calendars"
    ) from exc


# ---------------------------------------------------------------------
# Logging (uniforme)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_01")


# ---------------------------------------------------------------------
# Configuración de rutas (DVC-friendly) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
IN_RAW_PARQUET = Path(os.environ.get("IN_RAW_PARQUET", "data/raw/mnq_raw.parquet"))
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET", "data/processed/mnq_intraday.parquet"))
REPORT_SUMMARY = Path(
    os.environ.get("REPORT_SUMMARY", "reports/stage_01_dataset_prep_summary.json")
)


# ---------------------------------------------------------------------
# Configuración funcional (params reproducibles) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
STAGE_NAME = "stage_01_intraday_data_preparation"
STAGE_VERSION = "1.0"

MARKET = os.environ.get("MARKET", "NASDAQ")
TRADING_START = os.environ.get("TRADING_START", "06:30:00")
TRADING_END = os.environ.get("TRADING_END", "16:00:00")
TZ_FROM = os.environ.get("TZ_FROM", "UTC")
TZ_TO = os.environ.get("TZ_TO", "America/New_York")
GAP_MINUTES = int(os.environ.get("GAP_MINUTES", "1"))

ENABLE_MLFLOW = os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "yes", "YES"}


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
        "paths": {
            "inputs": inputs,
            "outputs": outputs,
            "reports": reports,
        },
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

    print("[PATHS]")
    paths = summary.get("paths", {}) or {}
    for group in ("inputs", "outputs", "reports"):
        print(f"  {group}:")
        for k, v in (paths.get(group, {}) or {}).items():
            print(f"    - {k}: {v}")

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
    MLflow opcional. Si enable=False, no hace nada.
    Loguea:
      - params: summary["params"]
      - metrics: summary["metrics"]
      - artifact: JSON summary
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
            if isinstance(v, (int, float)) and v == v:  # evita NaN
                mlflow.log_metric(k, float(v))

        mlflow.log_artifact(str(summary_path))


# ---------------------------------------------------------------------
# Core stage functions (puras)
# ---------------------------------------------------------------------
def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """Carga raw parquet y asegura DatetimeIndex ordenado."""
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {raw_path}")

    df = pd.read_parquet(raw_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise TypeError("El DataFrame debe tener DatetimeIndex o una columna 'datetime'.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def count_total_days(df: pd.DataFrame) -> int:
    return int(df.index.normalize().nunique())


def configure_timezone(df: pd.DataFrame, *, from_tz: str, to_tz: str) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(from_tz)
    df.index = df.index.tz_convert(to_tz)
    return df


def filter_market_trading_days(df: pd.DataFrame, *, market: str) -> pd.DataFrame:
    cal = mcal.get_calendar(market)
    start_date = df.index.min().date()
    end_date = df.index.max().date()

    trading_days = cal.schedule(start_date=start_date, end_date=end_date).index.normalize()
    idx_days = df.index.normalize().tz_localize(None)
    return df[idx_days.isin(trading_days)]


def filter_trading_hours(df: pd.DataFrame, *, start_time: str, end_time: str) -> pd.DataFrame:
    return df.between_time(start_time, end_time)


def analyze_daily_record_counts(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    daily_counts = df.groupby(df.index.date).size()
    full_day_record_count = int(daily_counts.mode().iloc[0]) if len(daily_counts) else 0
    return daily_counts, full_day_record_count


def find_incomplete_trading_dates(df: pd.DataFrame, *, expected_records: int, gap_minutes: int) -> List:
    """
    Marca un día como problemático si:
    - record_count < expected_records, o
    - existe algún gap intradiario != expected (excluye primer diff del día)
    """
    df = df.copy()
    df["time_diff"] = df.index.to_series().diff()
    expected_time_diff = pd.Timedelta(minutes=gap_minutes)

    daily_counts = df.groupby(df.index.date).size()
    problematic_dates: List = []

    for date, group in df.groupby(df.index.date):
        time_diffs = group["time_diff"].iloc[1:]
        has_irregular_gaps = (time_diffs != expected_time_diff).any()
        record_count = int(daily_counts[date])

        if (record_count < expected_records) or has_irregular_gaps:
            problematic_dates.append(date)

    return problematic_dates


def remove_incomplete_trading_days(df: pd.DataFrame, *, expected_records: int, gap_minutes: int) -> pd.DataFrame:
    problematic_dates = find_incomplete_trading_dates(
        df, expected_records=expected_records, gap_minutes=gap_minutes
    )
    return df[~df.index.to_series().dt.date.isin(problematic_dates)]


def get_trading_time_range(df: pd.DataFrame) -> Tuple[str, str]:
    times = df.index.time
    start_time = min(times).strftime("%H:%M:%S")
    end_time = max(times).strftime("%H:%M:%S")
    return start_time, end_time


def compute_stage_01_metrics_details(
    *,
    total_days_raw: int,
    daily_counts_clean: pd.Series,
    full_day_record_count_clean: int,
    total_nans: int,
    start_time_effective: str,
    end_time_effective: str,
    trading_hours_label: str,
    n_days_with_gaps: int,
    example_gap_dates: List,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    trading_days_output = int(daily_counts_clean.shape[0])

    discarded_days = int(total_days_raw - trading_days_output)
    discarded_days_pct = (discarded_days / total_days_raw) * 100.0 if total_days_raw else 0.0

    rpd_min = int(daily_counts_clean.min()) if len(daily_counts_clean) else 0
    rpd_median = float(daily_counts_clean.median()) if len(daily_counts_clean) else 0.0
    rpd_max = int(daily_counts_clean.max()) if len(daily_counts_clean) else 0

    metrics: Dict[str, float] = {
        "total_days_raw": float(total_days_raw),
        "trading_days_output": float(trading_days_output),
        "discarded_days": float(discarded_days),
        "discarded_days_pct": float(round(discarded_days_pct, 6)),
        "expected_records_per_day": float(full_day_record_count_clean),
        "records_per_day_min": float(rpd_min),
        "records_per_day_median": float(rpd_median),
        "records_per_day_max": float(rpd_max),
        "total_nans": float(total_nans),
        "days_with_gaps_post_clean": float(n_days_with_gaps),
    }

    details: Dict[str, Any] = {
        "output": {
            "trading_session_complete_days": trading_days_output,
            "expected_records_per_day": full_day_record_count_clean,
            "trading_time_range_effective": {
                "start_time": start_time_effective,
                "end_time": end_time_effective,
            },
        },
        "records_per_day": {
            "min": rpd_min,
            "median": rpd_median,
            "max": rpd_max,
        },
        "quality_checks": {
            "total_nans": total_nans,
            "discarded_days": discarded_days,
            "discarded_days_pct": round(discarded_days_pct, 6),
            "days_with_gaps_post_clean": n_days_with_gaps,
            "example_gap_dates_post_clean": example_gap_dates,
        },
        "metadata": {
            "trading_hours_label": trading_hours_label,
        },
    }

    return metrics, details


# ---------------------------------------------------------------------
# Checks de integridad temporal (función reusable)
# ---------------------------------------------------------------------
def log_time_index_health(df: pd.DataFrame, label: str) -> None:
    is_dtindex = isinstance(df.index, pd.DatetimeIndex)
    is_monotonic = bool(df.index.is_monotonic_increasing) if is_dtindex else False
    n_duplicates = int(df.index.duplicated().sum()) if is_dtindex else -1
    tzinfo = str(df.index.tz) if (is_dtindex and df.index.tz is not None) else "None"

    log.info("[CHECK:%s] DatetimeIndex: %s", label, is_dtindex)
    log.info("[CHECK:%s] TZ: %s", label, tzinfo)
    log.info("[CHECK:%s] Monotonic increasing: %s", label, is_monotonic)
    log.info("[CHECK:%s] Duplicated timestamps: %s", label, n_duplicates)

    if not is_dtindex:
        log.warning("[WARN:%s] El índice NO es DatetimeIndex.", label)
    if is_dtindex and not is_monotonic:
        log.warning("[WARN:%s] El índice temporal NO es monótono creciente.", label)
    if is_dtindex and n_duplicates > 0:
        log.warning("[WARN:%s] Hay %s timestamps duplicados.", label, n_duplicates)


# ---------------------------------------------------------------------
# CLI (argparse) - override (no inventa defaults)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage_01: Intraday data preparation (MNQ).")

    parser.add_argument("--in-raw-parquet", type=str, default=str(IN_RAW_PARQUET))
    parser.add_argument("--out-parquet", type=str, default=str(OUT_PARQUET))
    parser.add_argument("--report-summary", type=str, default=str(REPORT_SUMMARY))

    parser.add_argument("--market", type=str, default=MARKET)
    parser.add_argument("--trading-start", type=str, default=TRADING_START)
    parser.add_argument("--trading-end", type=str, default=TRADING_END)
    parser.add_argument("--tz-from", type=str, default=TZ_FROM)
    parser.add_argument("--tz-to", type=str, default=TZ_TO)
    parser.add_argument("--gap-minutes", type=int, default=int(GAP_MINUTES))

    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=ENABLE_MLFLOW,
        help="Si se activa, registra params/métricas en MLflow",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main (orquestación)
# ---------------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_raw = Path(args.in_raw_parquet)
    out_parquet = Path(args.out_parquet)
    report_summary = Path(args.report_summary)

    params: Dict[str, Any] = {
        "market": args.market,
        "trading_start": args.trading_start,
        "trading_end": args.trading_end,
        "tz_from": args.tz_from,
        "tz_to": args.tz_to,
        "gap_minutes": int(args.gap_minutes),
        "in_raw_parquet": str(in_raw.as_posix()),
        "out_parquet": str(out_parquet.as_posix()),
        "report_summary": str(report_summary.as_posix()),
    }

    log.info("[1] Cargando raw parquet y asegurando DatetimeIndex")
    mnq_raw = load_raw_dataset(in_raw)
    log_time_index_health(mnq_raw, "raw_loaded")

    log.info("[2] Contando días totales en raw")
    total_days_raw = count_total_days(mnq_raw)

    log.info("[3] Normalizando timezone: from=%s -> to=%s", args.tz_from, args.tz_to)
    mnq_raw_tz = configure_timezone(mnq_raw, from_tz=args.tz_from, to_tz=args.tz_to)
    log_time_index_health(mnq_raw_tz, "raw_tz")

    log.info("[4] Filtrando días hábiles (%s) y horario (%s-%s)", args.market, args.trading_start, args.trading_end)
    mnq_trading_days = filter_market_trading_days(mnq_raw_tz, market=args.market)
    mnq_intraday = filter_trading_hours(mnq_trading_days, start_time=args.trading_start, end_time=args.trading_end)
    log_time_index_health(mnq_intraday, "intraday_pre_clean")

    log.info("[5] Conteo registros/día (pre-limpieza) y día completo (moda)")
    daily_counts, full_day_record_count = analyze_daily_record_counts(mnq_intraday)
    log.info("[5.1] Expected records/day (mode): %s", full_day_record_count)

    log.info("[6] Removiendo días incompletos / gaps (gap_minutes=%s)", args.gap_minutes)
    mnq_intraday_clean = remove_incomplete_trading_days(
        mnq_intraday,
        expected_records=full_day_record_count,
        gap_minutes=int(args.gap_minutes),
    )
    log_time_index_health(mnq_intraday_clean, "intraday_post_clean")

    log.info("[6.1] Re-analizando registros/día (post-limpieza)")
    daily_counts_clean, full_day_record_count_clean = analyze_daily_record_counts(mnq_intraday_clean)
    log.info("[6.2] Expected records/day (mode, clean): %s", full_day_record_count_clean)

    log.info("[7] NaNs totales del dataset final")
    total_nans = int(mnq_intraday_clean.isna().sum().sum())

    # Chequeo adicional: continuidad intradiaria por día (gap exacto) post-clean
    expected = pd.Timedelta(minutes=int(args.gap_minutes))
    gaps_by_day = (
        mnq_intraday_clean.index.to_series()
        .groupby(mnq_intraday_clean.index.date)
        .apply(lambda s: (s.diff().iloc[1:] != expected).any())
    )
    n_days_with_gaps = int(gaps_by_day.sum()) if len(gaps_by_day) else 0
    example_gap_dates = list(gaps_by_day[gaps_by_day].index[:10]) if n_days_with_gaps > 0 else []
    log.info("[CHECK:post_clean] Days with intraday gaps (!=%s): %s", expected, n_days_with_gaps)
    if n_days_with_gaps > 0:
        log.warning("[WARN:post_clean] Ejemplos de días con gaps: %s", example_gap_dates)

    log.info("[8] Horario efectivo presente en dataset final")
    start_eff, end_eff = get_trading_time_range(mnq_intraday_clean)
    trading_hours_label = f"{args.trading_start}-{args.trading_end} {args.tz_to}"

    log.info("[9] Guardando parquet procesado: %s", out_parquet)
    _ensure_parent_dir(out_parquet)
    mnq_intraday_clean.to_parquet(out_parquet, index=True)

    log.info("[10] Construyendo metrics/details y summary envelope")
    metrics, details = compute_stage_01_metrics_details(
        total_days_raw=total_days_raw,
        daily_counts_clean=daily_counts_clean,
        full_day_record_count_clean=full_day_record_count_clean,
        total_nans=total_nans,
        start_time_effective=start_eff,
        end_time_effective=end_eff,
        trading_hours_label=trading_hours_label,
        n_days_with_gaps=n_days_with_gaps,
        example_gap_dates=example_gap_dates,
    )

    summary = build_summary_envelope(
        stage=STAGE_NAME,
        version=STAGE_VERSION,
        inputs={"mnq_raw_parquet": str(in_raw.as_posix())},
        outputs={"mnq_intraday_parquet": str(out_parquet.as_posix())},
        reports={"dataset_prep_summary": str(report_summary.as_posix())},
        params=params,
        metrics=metrics,
        details=details,
    )

    log.info("[11] Guardando report summary JSON: %s", report_summary)
    save_json(summary, report_summary)
    print_summary_console(summary)

    log.info("[12] MLflow tracking (enable=%s)", bool(args.enable_mlflow))
    mlflow_log_from_summary(
        enable=bool(args.enable_mlflow),
        stage=STAGE_NAME,
        summary=summary,
        summary_path=report_summary,
        tags={"pipeline": "neural_profit", "dataset": "MNQ"},
    )

    log.info("[OK] Stage_01 completado")
    log.info("Output parquet: %s", out_parquet)
    log.info("Report summary: %s", report_summary)


# ---------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
