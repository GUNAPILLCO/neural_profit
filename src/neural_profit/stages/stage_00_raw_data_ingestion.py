"""
stage_00_raw_data_ingestion
Ingesta y consolidación de datos crudos MNQ.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

SOURCE_GLOB = "data/source/*.txt"
OUT_PARQUET = "data/raw/mnq_raw.parquet"
OUT_SUMMARY = "reports/ingest_summary.json"

def read_one_txt(path: Path) -> pd.DataFrame:
    # Ajuste sep/cols según su formato real
    # Ejemplo típico: timestamp, open, high, low, close, volume
    df = pd.read_csv(
        path,
        sep=",",
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def main() -> None:
    files = sorted(Path("data/source").glob("*.txt"))
    if not files:
        raise SystemExit("ERROR: data/source está vacío. Copie los .txt allí antes de ejecutar DVC.")

    dfs = []
    for f in files:
        dfs.append(read_one_txt(f))

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Asegurar carpetas
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_SUMMARY).parent.mkdir(parents=True, exist_ok=True)

    # Guardar raw parquet
    df.to_parquet(OUT_PARQUET, index=False)

    # Summary (artefacto)
    summary = {
        "n_files": int(len(files)),
        "n_rows_total": int(len(df)),
        "min_timestamp": str(df["timestamp"].min()),
        "max_timestamp": str(df["timestamp"].max()),
        "columns": list(df.columns),
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("OK:", OUT_PARQUET)
    print("OK:", OUT_SUMMARY)

if __name__ == "__main__":
    main()