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
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/ingest_summary.json"))


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
# Función principal de ingesta (equivalente a su notebook generar_df)
# ---------------------------------------------------------------------
def generar_df(source_dir: Path) -> pd.DataFrame:
    """
    Lee todos los archivos .txt desde source_dir, los concatena,
    parsea el datetime y devuelve un DataFrame único indexado por datetime.
    """

    # Buscar archivos históricos
    files = sorted(source_dir.glob("*.txt"))

    # Falla explícita si no hay datos (mejor que un error silencioso)
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos históricos en: {source_dir}"
        )

    # Lista de DataFrames individuales
    dfs: list[pd.DataFrame] = []

    # Loop archivo por archivo (idéntico a su notebook)
    
    for archivo in files:

        # Leer el .txt crudo
        df = pd.read_csv(
            archivo,
            sep=SEP,
            header=None,
            names=COLS,
            dtype=DTYPES,
        )

        # Parseo estricto del datetime (sin warnings de pandas)
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            format=DATETIME_FORMAT,
            errors="coerce",
        )

        # Eliminar filas con datetime inválido (si existieran)
        df = df.dropna(subset=["datetime"])

        # Igual que en su notebook:
        # el datetime se convierte en índice temporal
        df = df.set_index("datetime")

        # Guardar DataFrame individual
        dfs.append(df)

    # Concatenar todos los archivos históricos
    df_mnq_raw = pd.concat(dfs)

    # Asegurar orden temporal creciente
    df_mnq_raw.sort_index(inplace=True)

    return df_mnq_raw


# ---------------------------------------------------------------------
# Artefacto: resumen de ingesta (metrics livianas)
# ---------------------------------------------------------------------
def write_ingest_summary(df: pd.DataFrame, n_files: int) -> dict:
    """
    Genera un resumen mínimo del proceso de ingesta.
    Este archivo se usa como:
    - métrica de DVC
    - artifact liviano de auditoría
    """

    summary = {
        "n_files": int(n_files),                  # cantidad de .txt ingeridos
        "n_rows_total": int(len(df)),             # total de filas
        "min_datetime": str(df.index.min()),      # inicio del rango temporal
        "max_datetime": str(df.index.max()),      # fin del rango temporal
        "columns": list(df.columns),              # columnas de datos
        "index_name": str(df.index.name),         # nombre del índice
    }

    # Crear carpeta reports/ si no existe
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    # Guardar JSON
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


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
    df_mnq_raw = generar_df(SOURCE_DIR)
    
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

    log.info(f"[4] Generando ingest_summary.json")
    # Generar artefacto de resumen
    summary = write_ingest_summary(
        df_mnq_raw,
        n_files=len(files),
    )

    # Logs simples (útiles en consola / CI)
    log.info(f"[OK] Raw parquet: {OUT_PARQUET}")
    log.info(f"[OK] Summary JSON: {OUT_SUMMARY}")

# ---------------------------------------------------------------------
# Boilerplate Python estándar
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
