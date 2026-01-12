"""
stage_01_intraday_data_preparation.py

Stage_01: Intraday dataset preparation for MNQ.

Outputs
-------
- data/processed/mnq_intraday.parquet        (DVC artifact)
- reports/dataset_prep_summary.json          (report artifact)
- MLflow params/metrics (optional)

Notes
-----
- Se asume que el input es un parquet minuto-a-minuto con OHLCV.
- Se usa calendario NASDAQ para filtrar días hábiles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("stage_01")


# Import del calendario de mercado.
# Si no está instalado, se falla temprano con un mensaje claro.
try:
    import pandas_market_calendars as mcal
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: pandas_market_calendars. "
        "Install with: pip install pandas-market-calendars"
    ) from exc


# ---------------------------------------------------------------------
# Utilidades del stage_01
# ---------------------------------------------------------------------
def load_raw_dataset(raw_path: Path) -> pd.DataFrame:
    """
    Carga el dataset raw desde parquet y valida que el índice sea DatetimeIndex.
    """
    # Validación defensiva: asegurar que el archivo exista.
    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {raw_path}")

    # Lectura del parquet raw.
    df = pd.read_parquet(raw_path)

    # Asegurar DatetimeIndex:
    # - Si el índice NO es DatetimeIndex, intentamos usar una columna 'datetime'.
    # - Si tampoco existe, se considera formato inválido.
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise TypeError("El DataFrame debe tener DatetimeIndex o una columna 'datetime'.")

    # Normalización: garantizar tipo datetime en el índice y orden cronológico.
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def count_total_days(df: pd.DataFrame) -> int:
    """
    Cuenta la cantidad total de días distintos presentes en un DataFrame
    con índice de tipo DatetimeIndex.
    """
    # normalize() trunca a 00:00:00, dejando solo la fecha; luego contamos únicos.
    return int(df.index.normalize().nunique())


def configure_timezone(
    df: pd.DataFrame,
    from_tz: str = "UTC",
    to_tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Asegura que el índice del DataFrame tenga definida la zona horaria `from_tz`
    y luego lo convierte a la zona horaria `to_tz`.

    Motivo:
    - Para filtrar correctamente el “horario de trading”, primero hay que estar en
      la zona horaria del mercado (típicamente America/New_York).
    """
    df = df.copy()

    # Si el índice viene "naive" (sin tz), lo localizamos en from_tz.
    # Nota: tz_localize NO cambia los timestamps; solo agrega tz info.
    if df.index.tz is None:
        df.index = df.index.tz_localize(from_tz)

    # Convertimos a la tz del mercado (esto sí puede mover horas).
    df.index = df.index.tz_convert(to_tz)
    return df


def filter_nasdaq_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra un DataFrame para conservar únicamente los días hábiles
    de negociación del mercado NASDAQ, en función de su DatetimeIndex.
    """
    nasdaq_calendar = mcal.get_calendar("NASDAQ")

    start_date = df.index.min().date()
    end_date = df.index.max().date()

    # Días hábiles oficiales del NASDAQ en el rango del dataset
    trading_days = nasdaq_calendar.schedule(
        start_date=start_date,
        end_date=end_date
    ).index.normalize()  # DatetimeIndex a medianoche (tz-naive)

    # Normalizar el índice del df a medianoche y quitar tz para comparar con trading_days
    idx_days = df.index.normalize().tz_localize(None)

    # Mantener solo filas cuya fecha esté dentro de los días hábiles
    filtered_df = df[idx_days.isin(trading_days)]

    return filtered_df


def filter_nasdaq_trading_hours(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Filtra un DataFrame para conservar únicamente las filas dentro del horario de trading.

    Motivo:
    - Aísla el “trading session” efectivo (ej. 06:30–16:00 NY),
      descartando pre/post market si no se desea.
    """
    return df.between_time(start_time, end_time)


def analyze_daily_record_counts(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Analiza la cantidad de registros por día.

    Retorna:
      - daily_counts: Serie con el conteo diario (por fecha).
      - full_day_record_count: cantidad de registros para un día “completo” (moda).
    """
    # Conteo de filas por día (agrupando por fecha).
    daily_counts = df.groupby(df.index.date).size()

    # “Día completo” se define como el valor modal (más frecuente) de registros/día.
    # Esto funciona bien cuando la mayoría de días están correctamente muestreados.
    full_day_record_count = int(daily_counts.mode().iloc[0]) if len(daily_counts) else 0
    return daily_counts, full_day_record_count


def find_incomplete_trading_dates(df: pd.DataFrame, expected_records: int, gap_minutes: int = 1) -> list:
    """
    Identifica fechas con:
    - menos registros que los esperados, o
    - irregularidades temporales (gaps) respecto del intervalo esperado.

    Retorna:
    - Lista de fechas (datetime.date) problemáticas.

    Motivo:
    - Detectar días con missing minutes, cortes de feed, o sesiones parciales.
    """
    df = df.copy()

    # Calcular la diferencia entre timestamps consecutivos (a nivel dataset completo).
    # Luego se analiza por día.
    df["time_diff"] = df.index.to_series().diff()

    # Intervalo esperado entre muestras (por defecto 1 minuto).
    expected_time_diff = pd.Timedelta(minutes=gap_minutes)

    # Conteo de filas por día.
    daily_counts = df.groupby(df.index.date).size()
    problematic_dates = []

    # Iterar por día para:
    # - revisar si hay gaps irregulares dentro del día
    # - revisar si faltan registros comparado al expected_records
    for date, group in df.groupby(df.index.date):
        # Se omite el primer registro del día porque su diff depende del día anterior.
        time_diffs = group["time_diff"].iloc[1:]

        # Hay irregularidades si algún diff es distinto al esperado.
        has_irregular_gaps = (time_diffs != expected_time_diff).any()

        # Registros reales del día.
        record_count = int(daily_counts[date])

        # Marcar el día como problemático si:
        # - tiene menos registros que expected_records, o
        # - tiene gaps irregulares
        if (record_count < expected_records) or has_irregular_gaps:
            problematic_dates.append(date)

    return problematic_dates


def remove_incomplete_trading_days(df: pd.DataFrame, expected_records: int, gap_minutes: int = 1) -> pd.DataFrame:
    """
    Elimina del DataFrame los días incompletos o con irregularidades temporales.

    Motivo:
    - Stage_01 busca producir un dataset intradía “limpio” y consistente por día.
    """
    # Identificar fechas problemáticas.
    problematic_dates = find_incomplete_trading_dates(
        df=df,
        expected_records=expected_records,
        gap_minutes=gap_minutes
    )

    # Eliminar esas fechas del dataset.
    cleaned_df = df[~df.index.to_series().dt.date.isin(problematic_dates)]
    return cleaned_df


def get_trading_time_range(df: pd.DataFrame) -> tuple[str, str]:
    """
    Obtiene la hora mínima y máxima presentes en el índice del DataFrame.

    Motivo:
    - Registrar “horario efectivo” del dataset final como evidencia/param (MLflow/JSON).
    """
    times = df.index.time
    start_time = min(times).strftime("%H:%M:%S")
    end_time = max(times).strftime("%H:%M:%S")
    return start_time, end_time


def build_dataset_prep_summary(
    *,
    total_days_raw: int,
    trading_days_output: int,
    records_per_day: pd.Series,
    full_day_record_count: int,
    total_nans: int,
    start_time: str,
    end_time: str,
    market: str,
    trading_hours_label: str, 
) -> Dict[str, Any]:
    """
    Construye el summary requerido para reports/dataset_prep_summary.json.

    Requeridos por usted:
    - #días (output)
    - %días descartados
    - registros/día (min/median/max)
    - #NaNs
    """
    # Cálculo de descartes (raw vs output)
    discarded_days = int(total_days_raw - trading_days_output)
    discarded_days_pct = round((discarded_days / total_days_raw) * 100, 4) if total_days_raw else 0.0

    # Estructura del JSON: separa input/output, stats y metadata.
    summary: Dict[str, Any] = {
        "input": {
            "total_days_raw": int(total_days_raw),
        },
        "output": {
            "trading_session_complete_days": int(trading_days_output),
            "expected_records_per_day": int(full_day_record_count),
            "trading_time_range": {"start_time": start_time, "end_time": end_time},
        },
        "records_per_day": {
            "min": int(records_per_day.min()) if len(records_per_day) else 0,
            "median": float(records_per_day.median()) if len(records_per_day) else 0.0,
            "max": int(records_per_day.max()) if len(records_per_day) else 0,
        },
        "quality_checks": {
            "total_nans": int(total_nans),
            "discarded_days": discarded_days,
            "discarded_days_pct": discarded_days_pct,
        },
        "metadata": {
            "market": market,
            "trading_hours": trading_hours_label,
        },
    }
    return summary


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    """
    Guarda un diccionario como JSON (UTF-8).

    Motivo:
    - Reporte reproducible y auditable del stage_01.
    """
    # Crear carpeta destino si no existe.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Guardar JSON legible (indent) y con UTF-8 (acentos).
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def log_mlflow(summary: Dict[str, Any], summary_path: Path, *, enable: bool) -> None:
    """
    Registra params/métricas en MLflow y adjunta el JSON como artefacto.
    Si enable=False, no hace nada.

    Motivo:
    - Trazabilidad: dejar registro de cómo se construyó el dataset procesado.
    """
    # Permite activar/desactivar MLflow sin tocar el código (por CLI/env var).
    if not enable:
        return

    # Import perezoso: solo requiere mlflow si el usuario lo habilita.
    try:
        import mlflow
    except ImportError:  # pragma: no cover
        raise ImportError("MLflow no está instalado. Instale con: pip install mlflow")

    # -----------------------
    # Params (configuración)
    # -----------------------
    mlflow.log_param("market", summary["metadata"]["market"])
    mlflow.log_param("trading_hours", summary["metadata"]["trading_hours"])
    mlflow.log_param("trading_start_time", summary["output"]["trading_time_range"]["start_time"])
    mlflow.log_param("trading_end_time", summary["output"]["trading_time_range"]["end_time"])

    # -----------------------
    # Metrics (numéricas)
    # -----------------------
    mlflow.log_metric("total_days_raw", float(summary["input"]["total_days_raw"]))
    mlflow.log_metric("trading_days", float(summary["output"]["trading_session_complete_days"]))
    mlflow.log_metric("discarded_days", float(summary["quality_checks"]["discarded_days"]))
    mlflow.log_metric("discarded_days_pct", float(summary["quality_checks"]["discarded_days_pct"]))
    mlflow.log_metric("records_per_day_min", float(summary["records_per_day"]["min"]))
    mlflow.log_metric("records_per_day_median", float(summary["records_per_day"]["median"]))
    mlflow.log_metric("records_per_day_max", float(summary["records_per_day"]["max"]))
    mlflow.log_metric("expected_records_per_day", float(summary["output"]["expected_records_per_day"]))
    mlflow.log_metric("total_nans", float(summary["quality_checks"]["total_nans"]))

    # -----------------------
    # Artifact
    # -----------------------
    # Adjuntamos el JSON para auditoría: permite inspección posterior del run.
    mlflow.log_artifact(str(summary_path))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Parseo de argumentos del script.
    Se pueden pasar por CLI o por variables de entorno.

    Motivo:
    - Integración fácil con DVC (dvc.yaml) y ejecución reproducible.
    """
    parser = argparse.ArgumentParser(description="Stage_01: Intraday data preparation (MNQ).")

    # Input raw (dep DVC)
    parser.add_argument(
        "--raw-path",
        type=str,
        default=os.environ.get("RAW_DIR", "data/raw/mnq_raw.parquet"),
        help="Ruta al parquet raw (default: data/raw/mnq_raw.parquet)",
    )

    # Output parquet procesado (out DVC)
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=os.environ.get("OUT_PARQUET", "data/processed/mnq_intraday.parquet"),
        help="Ruta del parquet procesado (default: data/processed/mnq_intraday.parquet)",
    )

    # Output report JSON (puede ir como metrics: en DVC)
    parser.add_argument(
        "--out-summary",
        type=str,
        default=os.environ.get("OUT_SUMMARY", "reports/dataset_prep_summary.json"),
        help="Ruta del JSON summary (default: reports/dataset_prep_summary.json)",
    )

    # Config de mercado/horario (para reproducibilidad y params MLflow)
    parser.add_argument("--market", type=str, default=os.environ.get("MARKET", "NASDAQ"))
    parser.add_argument("--trading-start", type=str, default=os.environ.get("TRADING_START", "06:30:00"))
    parser.add_argument("--trading-end", type=str, default=os.environ.get("TRADING_END", "16:00:00"))
    parser.add_argument("--timezone-from", type=str, default=os.environ.get("TZ_FROM", "UTC"))
    parser.add_argument("--timezone-to", type=str, default=os.environ.get("TZ_TO", "America/New_York"))

    # Validación de continuidad temporal (por defecto: 1 minuto)
    parser.add_argument("--gap-minutes", type=int, default=int(os.environ.get("GAP_MINUTES", "1")))

    # Toggle para MLflow (sin obligarlo)
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "YES", "yes"},
        help="Si se activa, registra params/métricas en MLflow",
    )

    return parser.parse_args()


def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    raw_path = Path(args.raw_path)
    out_parquet = Path(args.out_parquet)
    out_summary = Path(args.out_summary)

    log.info("[1] Cargando raw parquet y asegurando DatetimeIndex")
    mnq_raw = load_raw_dataset(raw_path)

    log.info("[1.1] Contando días totales en raw")
    total_days_raw = count_total_days(mnq_raw)

    log.info("[2] Normalizando timezone: from=%s -> to=%s", args.timezone_from, args.timezone_to)
    mnq_raw_tz = configure_timezone(mnq_raw, from_tz=args.timezone_from, to_tz=args.timezone_to)

    log.info("[3] Filtrando días hábiles NASDAQ y horario de trading (%s-%s)", args.trading_start, args.trading_end)
    mnq_trading_days = filter_nasdaq_trading_days(mnq_raw_tz)
    mnq_intraday = filter_nasdaq_trading_hours(mnq_trading_days, args.trading_start, args.trading_end)

    log.info("[4] Analizando conteo de registros por día (pre-limpieza)")
    daily_counts, full_day_record_count = analyze_daily_record_counts(mnq_intraday)
    log.info("[4.1] Registros por día (moda día completo): %s", full_day_record_count)

    log.info("[5] Removiendo días incompletos / gaps (gap_minutes=%s)", args.gap_minutes)
    mnq_intraday_clean = remove_incomplete_trading_days(
        mnq_intraday,
        expected_records=full_day_record_count,
        gap_minutes=args.gap_minutes,
    )

    log.info("[5.1] Re-analizando conteo de registros por día (post-limpieza)")
    daily_counts_clean, full_day_record_count_clean = analyze_daily_record_counts(mnq_intraday_clean)
    log.info("[5.2] Registros por día (moda día completo, limpio): %s", full_day_record_count_clean)

    log.info("[6] Calculando NaNs totales del dataset final")
    total_nans = int(mnq_intraday_clean.isna().sum().sum())
    log.info("[6.1] NaNs total: %s", total_nans)

    log.info("[7] Calculando horario efectivo presente en dataset final")
    start_time, end_time = get_trading_time_range(mnq_intraday_clean)
    trading_hours_label = f"{args.trading_start}-{args.trading_end} {args.timezone_to}"
    log.info("[7.1] Time range efectivo: %s -> %s (%s)", start_time, end_time, trading_hours_label)

    log.info("[8] Construyendo y guardando summary JSON: %s", out_summary)
    trading_days_output = int(daily_counts_clean.shape[0])
    summary = build_dataset_prep_summary(
        total_days_raw=total_days_raw,
        trading_days_output=trading_days_output,
        records_per_day=daily_counts_clean,
        full_day_record_count=full_day_record_count_clean,
        total_nans=total_nans,
        start_time=start_time,
        end_time=end_time,
        market=args.market,
        trading_hours_label=trading_hours_label,
    )
    save_json(summary, out_summary)

    log.info("[9] Guardando parquet procesado: %s", out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    mnq_intraday_clean.to_parquet(out_parquet, index=True)

    log.info("[10] Logging opcional a MLflow (enable=%s)", args.enable_mlflow)
    log_mlflow(summary, out_summary, enable=args.enable_mlflow)

    # Resumen final (sin prints)
    log.info("[OK] Stage_01 completed")
    log.info("Input days (raw): %s", total_days_raw)
    log.info("Output trading days: %s", trading_days_output)
    log.info("Discarded days (%%): %s", summary["quality_checks"]["discarded_days_pct"])
    log.info(
        "Records/day (min/median/max): %s/%s/%s",
        summary["records_per_day"]["min"],
        summary["records_per_day"]["median"],
        summary["records_per_day"]["max"],
    )
    log.info("NaNs total: %s", total_nans)
    log.info("Output parquet: %s", out_parquet)
    log.info("Summary JSON: %s", out_summary)

if __name__ == "__main__":
    # Entry point estándar: permite ejecutar el script por CLI o desde DVC.
    main()
