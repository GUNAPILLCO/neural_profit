"""
stage_00_raw_data_ingestion.py

Contrato del Stage
------------------
Propósito:
- Ingesta y consolidación de archivos históricos crudos de MNQ (.txt) en un único
  dataset estructurado, SIN aplicar filtros ni transformaciones de negocio.

Inputs (deps DVC):
- data/source/*.txt

Outputs (outs DVC):
- data/raw/mnq_raw.parquet

Reports (auditoría):
- reports/stage_00_ingest_summary.json

Params:
- SOURCE_DIR, OUT_PARQUET, REPORT_SUMMARY (paths DVC-friendly; env-first; CLI override)
- DATETIME_FORMAT, SEP, DTYPES, COLS (formato de los .txt)

Notas:
- El report JSON (summary envelope) es la fuente de verdad del stage.
- MLflow: se loguean params/metrics y se adjunta el JSON como artifact (REQUERIDO en este stage).
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Imports (stdlib)
# ---------------------------------------------------------------------
import argparse
import hashlib
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


# ---------------------------------------------------------------------
# Logging (uniforme)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_00")


# ---------------------------------------------------------------------
# Configuración de rutas (DVC-friendly) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
IN_SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", "data/source"))
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET", "data/raw/mnq_raw.parquet"))
REPORT_SUMMARY = Path(os.environ.get("REPORT_SUMMARY", "reports/stage_00_ingest_summary.json"))


# ---------------------------------------------------------------------
# Configuración funcional (params reproducibles) - SIEMPRE PRESENTE
# ---------------------------------------------------------------------
STAGE_NAME = "stage_00_raw_data_ingestion"
STAGE_VERSION = "1.0"

DATETIME_FORMAT = os.environ.get("DATETIME_FORMAT", "%Y%m%d %H%M%S")
SEP = os.environ.get("SEP", ";")

# Tipos de datos explícitos (evita inferencias inconsistentes)
DTYPES: Dict[str, str] = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}

# Orden y nombres esperados de las columnas en los .txt
COLS: List[str] = ["datetime", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------
# Utilidades generales (reusables)
# ---------------------------------------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


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
    stage: str,
    summary: Dict[str, Any],
    summary_path: Path,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    MLflow REQUIRED en este stage: si no está instalado, el stage falla.
    Loguea:
      - params: summary["params"]
      - metrics: summary["metrics"]
      - artifact: JSON summary
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow es requerido en stage_00. Instale con: pip install mlflow") from exc

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
def generate_df(
    source_dir: Path,
    *,
    sep: str,
    datetime_format: str,
    cols: List[str],
    dtypes: Dict[str, str],
) -> Tuple[pd.DataFrame, List[Path], Dict[str, int], Dict[str, int], int]:
    """
    Lee todos los archivos .txt desde source_dir, concatena, parsea datetime y devuelve:

    - df_mnq_raw (DatetimeIndex)
    - files (Paths ingeridos)
    - per_file_rows_read
    - per_file_invalid_datetime_dropped
    - invalid_datetime_dropped_total
    """
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos históricos en: {source_dir}")

    dfs: List[pd.DataFrame] = []
    per_file_rows_read: Dict[str, int] = {}
    per_file_invalid_datetime_dropped: Dict[str, int] = {}
    invalid_datetime_dropped_total: int = 0

    for archivo in files:
        df = pd.read_csv(
            archivo,
            sep=sep,
            header=None,
            names=cols,
            dtype=dtypes,
        )

        rows_read = int(len(df))

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format=datetime_format,
            errors="coerce",
        )

        invalid_dt = int(df["datetime"].isna().sum())
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime")

        per_file_rows_read[archivo.name] = rows_read
        per_file_invalid_datetime_dropped[archivo.name] = invalid_dt
        invalid_datetime_dropped_total += invalid_dt

        dfs.append(df)

    df_mnq_raw = pd.concat(dfs)
    df_mnq_raw.sort_index(inplace=True)

    # Validación mínima: índice temporal
    if not isinstance(df_mnq_raw.index, pd.DatetimeIndex):
        raise TypeError("El dataset consolidado no tiene DatetimeIndex tras la ingesta.")

    return (
        df_mnq_raw,
        files,
        per_file_rows_read,
        per_file_invalid_datetime_dropped,
        invalid_datetime_dropped_total,
    )


def compute_stage_00_metrics_details(
    df: pd.DataFrame,
    files: List[Path],
    *,
    source_dir: Path,
    out_parquet: Path,
    report_summary: Path,
    sep: str,
    datetime_format: str,
    cols: List[str],
    per_file_rows_read: Dict[str, int],
    per_file_invalid_datetime_dropped: Dict[str, int],
    invalid_datetime_dropped_total: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Calcula metrics (numéricas) y details (libre) para el summary envelope.
    """
    idx = df.index
    n_rows_total = int(len(df))

    is_monotonic = bool(idx.is_monotonic_increasing)
    n_duplicates = int(idx.duplicated().sum())

    min_dt = idx.min()
    max_dt = idx.max()

    n_days = int(pd.Series(idx.date).nunique()) if n_rows_total else 0

    # Timing diagnostics
    median_step_seconds = None
    p95_step_seconds = None
    max_gap_seconds = None
    pct_step_60s = None
    if n_rows_total >= 2:
        diffs = pd.Series(idx).diff().dropna()
        diffs_s = diffs.dt.total_seconds().astype("float64")
        median_step_seconds = float(diffs_s.median())
        p95_step_seconds = float(diffs_s.quantile(0.95))
        max_gap_seconds = float(diffs_s.max())
        pct_step_60s = float((diffs_s == 60.0).mean())

    # Data quality
    null_ratio_by_col: Dict[str, float] = {c: float(df[c].isna().mean()) for c in df.columns}

    col_minmax: Dict[str, Dict[str, Optional[float]]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            mn = df[c].min(skipna=True)
            mx = df[c].max(skipna=True)
            col_minmax[c] = {
                "min": None if pd.isna(mn) else float(mn),
                "max": None if pd.isna(mx) else float(mx),
            }
        else:
            col_minmax[c] = {"min": None, "max": None}

    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    non_positive_prices_count = int((df[price_cols] <= 0).any(axis=1).sum()) if price_cols else 0

    negative_volume_count = 0
    if "volume" in df.columns and pd.api.types.is_numeric_dtype(df["volume"]):
        negative_volume_count = int((df["volume"] < 0).sum())

    # Per-file summaries
    file_summaries: List[Dict[str, Any]] = []
    for p in files:
        name = p.name
        entry = {
            "name": name,
            "path": str(p.as_posix()),
            "size_bytes": int(p.stat().st_size) if p.exists() else 0,
            "sha256": _sha256_file(p) if p.exists() else None,
            "rows_read": int(per_file_rows_read.get(name, 0)),
            "invalid_datetime_dropped": int(per_file_invalid_datetime_dropped.get(name, 0)),
        }
        file_summaries.append(entry)

    # ---- params (config reproducible; lo arma main, aquí solo details/metrics) ----
    metrics: Dict[str, float] = {
        "n_files": float(len(files)),
        "n_rows_total": float(n_rows_total),
        "n_days": float(n_days),
        "invalid_datetime_dropped_total": float(invalid_datetime_dropped_total),
        "index_is_monotonic_increasing": float(1.0 if is_monotonic else 0.0),
        "n_duplicates": float(n_duplicates),
        "median_step_seconds": float(median_step_seconds) if median_step_seconds is not None else float("nan"),
        "p95_step_seconds": float(p95_step_seconds) if p95_step_seconds is not None else float("nan"),
        "max_gap_seconds": float(max_gap_seconds) if max_gap_seconds is not None else float("nan"),
        "pct_step_60s": float(pct_step_60s) if pct_step_60s is not None else float("nan"),
        "non_positive_prices_count": float(non_positive_prices_count),
        "negative_volume_count": float(negative_volume_count),
    }

    details: Dict[str, Any] = {
        "time_coverage": {
            "min_datetime": str(min_dt) if min_dt is not pd.NaT else None,
            "max_datetime": str(max_dt) if max_dt is not pd.NaT else None,
        },
        "index_integrity": {
            "is_monotonic_increasing": is_monotonic,
            "n_duplicates": n_duplicates,
        },
        "raw_format": {
            "datetime_format": datetime_format,
            "sep": sep,
            "expected_columns": cols,
            "output_columns": list(df.columns),
            "index_name": str(df.index.name),
        },
        "timing_diagnostics": {
            "median_step_seconds": median_step_seconds,
            "p95_step_seconds": p95_step_seconds,
            "max_gap_seconds": max_gap_seconds,
            "pct_step_60s": pct_step_60s,
        },
        "data_quality": {
            "null_ratio_by_col": null_ratio_by_col,
            "col_minmax": col_minmax,
            "non_positive_prices_count": non_positive_prices_count,
            "negative_volume_count": negative_volume_count,
        },
        "files": file_summaries,
        "io": {
            "source_dir": str(source_dir.as_posix()),
            "out_parquet": str(out_parquet.as_posix()),
            "report_summary": str(report_summary.as_posix()),
        },
    }

    return metrics, details


# ---------------------------------------------------------------------
# CLI (argparse) - override (no inventa defaults)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage_00: Raw data ingestion (MNQ).")

    parser.add_argument("--source-dir", type=str, default=str(IN_SOURCE_DIR))
    parser.add_argument("--out-parquet", type=str, default=str(OUT_PARQUET))
    parser.add_argument("--report-summary", type=str, default=str(REPORT_SUMMARY))

    parser.add_argument("--datetime-format", type=str, default=DATETIME_FORMAT)
    parser.add_argument("--sep", type=str, default=SEP)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main (orquestación)
# ---------------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    source_dir = Path(args.source_dir)
    out_parquet = Path(args.out_parquet)
    report_summary = Path(args.report_summary)

    # Params efectivos usados (para summary + MLflow)
    params: Dict[str, Any] = {
        "source_dir": str(source_dir.as_posix()),
        "out_parquet": str(out_parquet.as_posix()),
        "report_summary": str(report_summary.as_posix()),
        "datetime_format": str(args.datetime_format),
        "sep": str(args.sep),
        "cols": list(COLS),
        "dtypes": dict(DTYPES),
    }

    log.info("[1] Verificando archivos .txt en: %s", source_dir)
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"ERROR: '{source_dir}' está vacío. Copie los .txt allí y reintente.")
    log.info("[OK] Archivos .txt encontrados: %s", len(files))

    log.info("[2] Ingestando y consolidando dataset RAW")
    (
        df_mnq_raw,
        files_used,
        per_file_rows_read,
        per_file_invalid_datetime_dropped,
        invalid_datetime_dropped_total,
    ) = generate_df(
        source_dir,
        sep=str(args.sep),
        datetime_format=str(args.datetime_format),
        cols=COLS,
        dtypes=DTYPES,
    )

    # Checks rápidos de integridad temporal (sin modificar)
    is_monotonic = bool(df_mnq_raw.index.is_monotonic_increasing)
    n_duplicates = int(df_mnq_raw.index.duplicated().sum())
    log.info("[CHECK] Index monotonic increasing: %s", is_monotonic)
    log.info("[CHECK] Duplicated timestamps: %s", n_duplicates)
    if not is_monotonic:
        log.warning("[WARN] Índice temporal NO monótono. Revisar solapes/orden de archivos.")
    if n_duplicates > 0:
        log.warning("[WARN] Timestamps duplicados detectados: %s", n_duplicates)

    log.info("[3] Guardando parquet RAW: %s", out_parquet)
    _ensure_parent_dir(out_parquet)
    df_mnq_raw.to_parquet(out_parquet, index=True)

    log.info("[4] Calculando metrics/details y construyendo summary envelope")
    metrics, details = compute_stage_00_metrics_details(
        df=df_mnq_raw,
        files=files_used,
        source_dir=source_dir,
        out_parquet=out_parquet,
        report_summary=report_summary,
        sep=str(args.sep),
        datetime_format=str(args.datetime_format),
        cols=COLS,
        per_file_rows_read=per_file_rows_read,
        per_file_invalid_datetime_dropped=per_file_invalid_datetime_dropped,
        invalid_datetime_dropped_total=invalid_datetime_dropped_total,
    )

    summary = build_summary_envelope(
        stage=STAGE_NAME,
        version=STAGE_VERSION,
        inputs={"source_dir": str(source_dir.as_posix())},
        outputs={"mnq_raw_parquet": str(out_parquet.as_posix())},
        reports={"ingest_summary": str(report_summary.as_posix())},
        params=params,
        metrics=metrics,
        details=details,
    )

    log.info("[5] Guardando report summary JSON: %s", report_summary)
    save_json(summary, report_summary)

    print_summary_console(summary)

    log.info("[6] MLflow tracking (REQUERIDO)")
    mlflow_log_from_summary(
        stage=STAGE_NAME,
        summary=summary,
        summary_path=report_summary,
        tags={"pipeline": "neural_profit", "dataset": "MNQ"},
    )

    log.info("[OK] Stage_00 completado")
    log.info("Output parquet: %s", out_parquet)
    log.info("Report summary: %s", report_summary)


# ---------------------------------------------------------------------
# Boilerplate
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
