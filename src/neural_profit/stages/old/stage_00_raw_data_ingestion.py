"""
stage_00_raw_data_ingestion

Descripción:
Ingesta y consolidación de los archivos históricos crudos de MNQ
en un único dataset estructurado, SIN aplicar filtros ni transformaciones
de negocio (horarios, calendarios, limpieza intradía, etc.).

Este stage define el "raw truth" del proyecto.

Input:
- data/source/*.txt   (archivos históricos crudos)

Output:
- data/raw/mnq_raw.parquet

Artefactos:
- reports/ingest_summary.json
"""

# ---------------------------------------------------------------------
# Imports estándar
# ---------------------------------------------------------------------

from __future__ import annotations  # permite type hints modernos (Python < 3.11)
import json                         # para guardar el summary como JSON
import os                           # para leer variables de entorno
from pathlib import Path            # manejo robusto de rutas
import pandas as pd                 # procesamiento de datos
import hashlib
from datetime import datetime, timezone
import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("stage_00")

# ---------------------------------------------------------------------
# Configuración de rutas (DVC-friendly)
# ---------------------------------------------------------------------
# Estas rutas son RELATIVAS al repositorio.
# DVC necesita paths locales y determinísticos.
# Si mañana quiere apuntar a Drive, se hace vía DVC remote o symlink,
# NO cambiando la lógica del stage.

SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", "data/source"))
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET", "data/raw/mnq_raw.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/stage_00_ingest_summary.json"))


# ---------------------------------------------------------------------
# Configuración del formato de los datos crudos
# ---------------------------------------------------------------------

# Formato exacto del datetime según sus .txt
# Ejemplo: 20200102 083000
DATETIME_FORMAT = "%Y%m%d %H%M%S"

# Separador de los archivos .txt
SEP = ";"

# Tipos de datos explícitos (evita inferencias inconsistentes)
DTYPES = {
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}

# Orden y nombres esperados de las columnas en los .txt
COLS = ["datetime", "open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------
# 1) Función principal de ingesta (equivalente a su notebook generar_df)
# ---------------------------------------------------------------------
def generate_df(
    source_dir: Path,
    ) -> tuple[pd.DataFrame, dict, dict, int]:
    """
    Lee todos los archivos .txt desde source_dir, los concatena,
    parsea el datetime y devuelve:

    - df_mnq_raw
    - per_file_rows_read
    - per_file_invalid_datetime_dropped
    - invalid_datetime_dropped_total
    """

    files = sorted(source_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos históricos en: {source_dir}"
        )

    dfs: list[pd.DataFrame] = []

    per_file_rows_read: dict[str, int] = {}
    per_file_invalid_datetime_dropped: dict[str, int] = {}
    invalid_datetime_dropped_total: int = 0

    for archivo in files:
        # Leer crudo
        df = pd.read_csv(
            archivo,
            sep=SEP,
            header=None,
            names=COLS,
            dtype=DTYPES,
        )

        rows_read = len(df)

        # Parseo datetime
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format=DATETIME_FORMAT,
            errors="coerce",
        )

        invalid_dt = int(df["datetime"].isna().sum())

        # Drop filas inválidas
        df = df.dropna(subset=["datetime"])

        # Set index temporal
        df = df.set_index("datetime")

        # Acumuladores
        per_file_rows_read[archivo.name] = rows_read
        per_file_invalid_datetime_dropped[archivo.name] = invalid_dt
        invalid_datetime_dropped_total += invalid_dt

        dfs.append(df)

    # Consolidación
    df_mnq_raw = pd.concat(dfs)
    df_mnq_raw.sort_index(inplace=True)

    return (
        df_mnq_raw,
        per_file_rows_read,
        per_file_invalid_datetime_dropped,
        invalid_datetime_dropped_total,
    )


# ---------------------------------------------------------------------
# Artefacto: resumen de ingesta (metrics livianas)
# ---------------------------------------------------------------------
def build_stage_00_summary(
    df: pd.DataFrame,
    files: List[Path],
    *,
    source_dir: Path,
    out_parquet: Path,
    out_summary: Path,
    datetime_format: str,
    sep: str,
    invalid_datetime_dropped_total: int = 0,
    per_file_rows_read: Optional[Dict[str, int]] = None,
    per_file_invalid_datetime_dropped: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Genera un summary enriquecido de la ingesta RAW (stage_00).

    - No aplica transformaciones de negocio.
    - El objetivo es auditoría / trazabilidad (metrics livianas).
    - Todo lo devuelto es JSON-serializable.

    Parámetros:
      df: DataFrame final consolidado (index datetime).
      files: lista de Paths a los .txt ingeridos (en orden).
      source_dir, out_parquet, out_summary: rutas usadas por el stage.
      datetime_format, sep: parámetros de parseo de crudos.
      invalid_datetime_dropped_total: conteo total de filas descartadas por datetime inválido.
      per_file_rows_read: dict opcional {filename: rows_leidas}
      per_file_invalid_datetime_dropped: dict opcional {filename: drops_datetime_invalido}

    Devuelve:
      summary: dict listo para guardarse como JSON.
    """

    def _iso_utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _safe_float(x) -> Optional[float]:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    def _safe_int(x) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    # -----------------------------
    # Checks básicos del índice
    # -----------------------------
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("build_stage_00_summary espera df.index como pd.DatetimeIndex")

    n_rows_total = int(len(df))
    min_dt = idx.min()
    max_dt = idx.max()

    is_monotonic = bool(idx.is_monotonic_increasing)
    n_duplicates = int(idx.duplicated().sum())

    # Cantidad de días (por fecha calendario) presentes en el RAW
    # (en RAW no asumimos sesión; es un conteo de fechas)
    n_days = int(pd.Series(idx.date).nunique()) if n_rows_total > 0 else 0

    # -----------------------------
    # Frecuencia / gaps (sin filtrar)
    # -----------------------------
    if n_rows_total >= 2:
        diffs = pd.Series(idx).diff().dropna()
        diffs_s = diffs.dt.total_seconds().astype("float64")

        median_step_seconds = _safe_float(diffs_s.median())
        p95_step_seconds = _safe_float(diffs_s.quantile(0.95))
        max_gap_seconds = _safe_float(diffs_s.max())

        # Un extra útil: % de pasos = 60s (si esperas 1-min)
        pct_step_60s = _safe_float((diffs_s == 60.0).mean())
    else:
        median_step_seconds = None
        p95_step_seconds = None
        max_gap_seconds = None
        pct_step_60s = None

    # -----------------------------
    # Calidad por columna (NaN ratios)
    # -----------------------------
    null_ratio_by_col: Dict[str, float] = {}
    for c in df.columns:
        null_ratio_by_col[c] = float(df[c].isna().mean()) if n_rows_total else 0.0

    # -----------------------------
    # Rangos por columna (min/max)
    # -----------------------------
    col_minmax: Dict[str, Dict[str, Optional[float]]] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            col_minmax[c] = {
                "min": _safe_float(df[c].min(skipna=True)),
                "max": _safe_float(df[c].max(skipna=True)),
            }
        else:
            col_minmax[c] = {"min": None, "max": None}

    # -----------------------------
    # Conteos de valores inválidos
    # -----------------------------
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    non_positive_prices_count = 0
    if price_cols:
        # Cuenta filas donde alguna price <= 0 (ignorando NaNs)
        non_positive_prices_count = int((df[price_cols] <= 0).any(axis=1).sum())

    negative_volume_count = 0
    if "volume" in df.columns and pd.api.types.is_numeric_dtype(df["volume"]):
        negative_volume_count = int((df["volume"] < 0).sum())

    # -----------------------------
    # Resumen por archivo
    # -----------------------------
    file_summaries: List[Dict[str, Any]] = []
    for p in files:
        name = p.name
        rows_read = per_file_rows_read.get(name) if per_file_rows_read else None
        drops_bad_dt = (
            per_file_invalid_datetime_dropped.get(name)
            if per_file_invalid_datetime_dropped
            else None
        )

        entry: Dict[str, Any] = {
            "name": name,
            "path": str(p.as_posix()),
            "size_bytes": _safe_int(p.stat().st_size) if p.exists() else 0,
            "sha256": _sha256_file(p) if p.exists() else None,
            "rows_read": rows_read,
            "invalid_datetime_dropped": drops_bad_dt,
        }
        file_summaries.append(entry)

    # -----------------------------
    # Summary final
    # -----------------------------
    summary: Dict[str, Any] = {
        "stage": "stage_00_raw_data_ingestion",
        "created_at_utc": _iso_utc_now(),
        "paths": {
            "source_dir": str(source_dir.as_posix()),
            "out_parquet": str(out_parquet.as_posix()),
            "out_summary": str(out_summary.as_posix()),
        },
        "raw_format": {
            "datetime_format": datetime_format,
            "sep": sep,
            "expected_columns": list(df.columns),
            "index_name": str(df.index.name),
        },
        "ingestion": {
            "n_files": int(len(files)),
            "n_rows_total": n_rows_total,
            "invalid_datetime_dropped_total": int(invalid_datetime_dropped_total),
        },
        "time_coverage": {
            "min_datetime": str(min_dt) if min_dt is not pd.NaT else None,
            "max_datetime": str(max_dt) if max_dt is not pd.NaT else None,
            "n_days": n_days,
        },
        "index_integrity": {
            "is_monotonic_increasing": is_monotonic,
            "n_duplicates": n_duplicates,
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
            "non_positive_prices_count": int(non_positive_prices_count),
            "negative_volume_count": int(negative_volume_count),
        },
        "files": file_summaries,
    }

    # Guardado (carpeta reports/)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_stage_00_summary_console(summary: dict) -> None:
    """
    Imprime en consola un resumen legible del stage_00_ingest_summary.json.
    Pensado para logs de terminal / CI.
    """

    def _fmt(x):
        return "N/A" if x is None else x

    print("\n" + "=" * 70)
    print(f"STAGE: {summary.get('stage')}")
    print(f"CREATED_AT (UTC): {summary.get('created_at_utc')}")
    print("=" * 70)

    # -----------------------------
    # Paths
    # -----------------------------
    paths = summary.get("paths", {})
    print("\n[PATHS]")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

    # -----------------------------
    # Ingestion
    # -----------------------------
    ingestion = summary.get("ingestion", {})
    print("\n[INGESTION]")
    print(f"  Files ingested        : {ingestion.get('n_files')}")
    print(f"  Total rows            : {ingestion.get('n_rows_total')}")
    print(
        f"  Invalid datetime drop : {ingestion.get('invalid_datetime_dropped_total')}"
    )

    # -----------------------------
    # Time coverage
    # -----------------------------
    time_cov = summary.get("time_coverage", {})
    print("\n[TIME COVERAGE]")
    print(f"  From : {time_cov.get('min_datetime')}")
    print(f"  To   : {time_cov.get('max_datetime')}")
    print(f"  Days : {time_cov.get('n_days')}")

    # -----------------------------
    # Index integrity
    # -----------------------------
    idx = summary.get("index_integrity", {})
    print("\n[INDEX INTEGRITY]")
    print(f"  Monotonic increasing : {idx.get('is_monotonic_increasing')}")
    print(f"  Duplicated timestamps: {idx.get('n_duplicates')}")

    # -----------------------------
    # Timing diagnostics
    # -----------------------------
    timing = summary.get("timing_diagnostics", {})
    print("\n[TIMING DIAGNOSTICS]")
    print(f"  Median step (s) : {_fmt(timing.get('median_step_seconds'))}")
    print(f"  P95 step (s)    : {_fmt(timing.get('p95_step_seconds'))}")
    print(f"  Max gap (s)     : {_fmt(timing.get('max_gap_seconds'))}")
    print(f"  % step = 60s    : {_fmt(timing.get('pct_step_60s'))}")

    # -----------------------------
    # Data quality
    # -----------------------------
    dq = summary.get("data_quality", {})
    print("\n[DATA QUALITY]")
    print("  Null ratio by column:")
    for col, ratio in dq.get("null_ratio_by_col", {}).items():
        print(f"    - {col:<6}: {ratio:.4%}")

    print(
        f"  Non-positive prices count : {dq.get('non_positive_prices_count')}"
    )
    print(
        f"  Negative volume count     : {dq.get('negative_volume_count')}"
    )

    # -----------------------------
    # Files
    # -----------------------------
    print("\n[FILES]")
    for f in summary.get("files", []):
        print(f"  - {f.get('name')}")
        #print(f"      rows_read                 : {_fmt(f.get('rows_read'))}")
        #print(
        #    f"      invalid_datetime_dropped  : {_fmt(f.get('invalid_datetime_dropped'))}"
        #)
        #print(f"      size_bytes                : {_fmt(f.get('size_bytes'))}")
        #print(f"      sha256                    : {_fmt(f.get('sha256'))}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------
# Entry point del stage (lo que ejecuta DVC)
# ---------------------------------------------------------------------
def main() -> None:
    """
    Punto de entrada del stage_00.
    DVC ejecuta esta función vía:
        dvc repro
    """

    # Verificación temprana: ¿hay archivos fuente?
    log.info(f"[1] Leyendo archivos .txt")
    
    files = sorted(SOURCE_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(
            f"ERROR: '{SOURCE_DIR}' está vacío. "
            "Copie los .txt allí y reintente."
        )
    
    log.info(f"[OK] Archivos .txt encontrados")
    # Construcción del dataset raw
    # Nota: en DVC normalmente se reconstruye siempre;
    # el control de cambios lo hace DVC con deps/outs.
    
    log.info(f"[2] Construyendo dataset mnq_raw.parquet")
    df_mnq_raw, per_file_rows_read, per_file_invalid_datetime_dropped, invalid_datetime_dropped_total = generate_df(SOURCE_DIR)
    log.info(f"[OK] Dataset construido correctamente")

    # Asegurar que exista data/raw/
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------------------
    # Checks de integridad temporal (NO modifican datos)
    # -----------------------------------------------------------------
    is_monotonic = df_mnq_raw.index.is_monotonic_increasing
    n_duplicates = int(df_mnq_raw.index.duplicated().sum())

    log.info(f"[CHECK] Index monotonic increasing: {is_monotonic}")
    log.info(f"[CHECK] Duplicated timestamps: {n_duplicates}")

    if not is_monotonic:
        log.warning(
            "[WARN] El índice temporal NO es monótono creciente. "
            "Revisar orden o solapes entre archivos."
        )

    if n_duplicates > 0:
        log.warning(
            f"[WARN] Se detectaron {n_duplicates} timestamps duplicados "
            "en el índice temporal."
        )  

    log.info(f"[3] Guardando dataset mnq_raw.parquet")
    # Guardar parquet RAW (con índice datetime)
    df_mnq_raw.to_parquet(OUT_PARQUET, index=True)
    log.info(f"[OK] Dataset PARQUET: {OUT_SUMMARY}")

    
    log.info(f"[4] Generando stage_00_ingest_summary.json")

    summary = build_stage_00_summary(
        df=df_mnq_raw,
        files=files,
        source_dir=SOURCE_DIR,
        out_parquet=OUT_PARQUET,
        out_summary=OUT_SUMMARY,  # ahora: reports/stage_00_ingest_summary.json
        datetime_format=DATETIME_FORMAT,
        sep=SEP,
        invalid_datetime_dropped_total=invalid_datetime_dropped_total,
        per_file_rows_read=per_file_rows_read,
        per_file_invalid_datetime_dropped=per_file_invalid_datetime_dropped,
    )

    log.info(f"[OK] Summary JSON: {OUT_SUMMARY}")
    print_stage_00_summary_console(summary)

    log.info("\n[OK] Stage_00 completo")
# ---------------------------------------------------------------------
# Boilerplate Python estándar
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
