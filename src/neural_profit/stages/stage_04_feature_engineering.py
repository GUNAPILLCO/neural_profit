# ============================================================
# stage_04_feature_engineering.py
# Feature Engineering (MNQ intraday): OHLC(+V opcional) + indicadores
# Guarda: data/features/mnq_features_target.parquet
# Reporte: reports/features_target_summary.json
# ============================================================

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from ta.momentum import ROCIndicator

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_04_feature_engineering")

# ----------------------------
# MLflow (opcional)
# ----------------------------
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None


# ============================================================
# Paths / IO (via env o defaults)
# ============================================================
IN_PARQUET = Path(os.environ.get("IN_PARQUET", "data/processed/mnq_intraday_labeled.parquet"))
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET", "data/features/mnq_features_target.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/features_target_summary.json"))

# Si en su pipeline hay un artifact de targets, puede quedar declarado,
# pero NO es estrictamente necesario en stage_04.
IN_ARTIFACT = Path(os.environ.get("IN_ARTIFACT", "reports/target_definition_summary.json"))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Carga y preprocesamiento base
# ============================================================
def load_mnq_parquet(path: Path = IN_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")
    log.info(f"[OK] Cargando parquet: {path}")
    return pd.read_parquet(path)


def add_column_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Asegura DatetimeIndex y agrega columna 'date' (YYYY-MM-DD) para agrupar por jornada.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out[date_col] = out.index.date
    # Reordenar (date primero)
    cols = [date_col] + [c for c in out.columns if c != date_col]
    return out[cols]


# ============================================================
# 2) Cálculo de indicadores técnicos por día (sin cruzar jornadas)
# ============================================================
def compute_selected_indicators_per_day(
    df: pd.DataFrame,
    target_col: str = "close",
    date_col: str = "date",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcula indicadores técnicos de forma independiente por día:
      - price_ema60  (close / EMA60 - 1)
      - momentum_10  (pct_change(10))
      - roc_30       (ROC window=30)
      - roc_60       (ROC window=60)

    Devuelve:
      df_out: DF con columnas nuevas
      indicator_columns: nombres de indicadores generados
    """

    indicator_columns: List[str] = [
        "price_ema60",
        "momentum_10",
        "roc_30",
        "roc_60",
    ]

    # Validaciones mínimas
    missing = [c for c in [target_col, date_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en df: {missing}")

    def apply_per_day(day_df: pd.DataFrame) -> pd.DataFrame:
        day_df = day_df.copy()

        # EMA-based price extension (normalized)
        log.info(f"[2.1] Calculando 'price_ema60'")
        day_df["price_ema60"] = day_df[target_col] / day_df[target_col].ewm(span=60).mean() - 1

        # Momentum (percentage change)
        (f"[2.2] Calculando 'momentum_10'")
        day_df["momentum_10"] = day_df[target_col].pct_change(10)

        # ROC 30 / 60
        (f"[2.3] Calculando 'roc_30'")
        day_df["roc_30"] = ROCIndicator(close=day_df[target_col], window=30).roc()
        
        (f"[2.4] Calculando 'roc_60'")
        day_df["roc_60"] = ROCIndicator(close=day_df[target_col], window=60).roc()

        return day_df

    df_out = df.groupby(date_col, group_keys=False).apply(apply_per_day)
    return df_out, indicator_columns


# ============================================================
# 3) Summary liviano (JSON) + Pretty print
# ============================================================
def build_stage04_summary_report(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    features: List[str],
    targets: List[str],
    date_col: str = "date",
    decimals: int = 6,
) -> Dict[str, Any]:
    """
    Summary del stage_04 (Feature Engineering):
      - tamaños antes/después
      - NaNs / filas descartadas
      - features/targets finales
      - días y filas por día

    Devuelve un dict serializable a JSON.
    """

    # Conteos base
    n_rows_in = int(df_in.shape[0])
    n_rows_out = int(df_out.shape[0])

    # Días
    n_days_in = int(df_in[date_col].nunique()) if date_col in df_in.columns else None
    n_days_out = int(df_out[date_col].nunique()) if date_col in df_out.columns else None

    # Filas por día (promedio)
    rows_per_day_in = float(df_in.groupby(date_col).size().mean()) if (date_col in df_in.columns and n_rows_in > 0) else None
    rows_per_day_out = float(df_out.groupby(date_col).size().mean()) if (date_col in df_out.columns and n_rows_out > 0) else None

    # NaN ratios (solo en columnas finales)
    final_cols = [date_col] + features + targets
    final_cols_existing = [c for c in final_cols if c in df_out.columns]

    nan_ratio_by_col = {}
    for c in final_cols_existing:
        nan_ratio_by_col[c] = float(df_out[c].isna().mean())

    # DropNA impacto (estimación a partir de df_out limpio vs pre-limpieza)
    # OJO: si df_out ya viene limpio, esto dará 0. Por eso lo pasamos con df_in/df_out "reales".
    dropped_rows = n_rows_in - n_rows_out  # si df_out es el dataset final limpio
    dropped_ratio = (dropped_rows / n_rows_in) if n_rows_in > 0 else 0.0

    summary = {
        "stage": "stage_04_feature_engineering",
        "description": "Construcción y consolidación de features intradía (OHLC + indicadores técnicos) y targets delta_pts_h.",
        "io": {
            "in_parquet": str(IN_PARQUET),
            "out_parquet": str(OUT_PARQUET),
            "out_summary": str(OUT_SUMMARY),
        },
        "schema": {
            "date_col": date_col,
            "features": features,
            "targets": targets,
            "n_features": int(len(features)),
            "n_targets": int(len(targets)),
        },
        "data_stats": {
            "n_rows_in": n_rows_in,
            "n_rows_out": n_rows_out,
            "rows_dropped": int(dropped_rows),
            "rows_dropped_ratio": round(float(dropped_ratio), 6),
            "n_days_in": n_days_in,
            "n_days_out": n_days_out,
            "rows_per_day_mean_in": None if rows_per_day_in is None else round(rows_per_day_in, 2),
            "rows_per_day_mean_out": None if rows_per_day_out is None else round(rows_per_day_out, 2),
        },
        "nan_ratio_by_col": {k: round(v, decimals) for k, v in nan_ratio_by_col.items()},
    }

    return summary


def print_stage04_summary_pretty(summary: Dict[str, Any]) -> None:
    """
    Pretty print de un summary dict de stage_04.
    """
    if not summary:
        print("Summary vacío (stage_04).")
        return

    ds = summary.get("data_stats", {})
    schema = summary.get("schema", {})

    print("\n" + "=" * 78)
    print("STAGE_04 – FEATURE ENGINEERING (MNQ)")
    print("=" * 78)

    print(f"\nFeatures finales ({schema.get('n_features', '?')}): {schema.get('features', [])}")
    print(f"Targets finales  ({schema.get('n_targets', '?')}): {schema.get('targets', [])}")

    print("\n" + "-" * 78)
    print("Data")
    print("-" * 78)
    print(f"Filas entrada : {ds.get('n_rows_in')}")
    print(f"Filas salida  : {ds.get('n_rows_out')}")
    print(f"Drop rows     : {ds.get('rows_dropped')}  (ratio={ds.get('rows_dropped_ratio')})")
    print(f"Días entrada  : {ds.get('n_days_in')}")
    print(f"Días salida   : {ds.get('n_days_out')}")
    print(f"Rows/day mean : in={ds.get('rows_per_day_mean_in')} | out={ds.get('rows_per_day_mean_out')}")

    print("\n" + "-" * 78)
    print("NaN ratio (cols finales)")
    print("-" * 78)
    nan_map = summary.get("nan_ratio_by_col", {})
    # Mostrar solo las más relevantes (si hay muchas)
    for k in sorted(nan_map.keys()):
        print(f"{k:<20} : {nan_map[k]}")

    print("\n" + "=" * 78)


# ============================================================
# 4) MLflow summary (params + metrics)
# ============================================================
def build_stage04_mlflow_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte el summary del stage_04 en un dict (params/metrics) para MLflow.
    """
    ds = summary.get("data_stats", {})
    schema = summary.get("schema", {})

    out = {
        "params": {
            "stage": summary.get("stage", "stage_04_feature_engineering"),
            "features": schema.get("features", []),
            "targets": schema.get("targets", []),
            "n_features": int(schema.get("n_features", 0)),
            "n_targets": int(schema.get("n_targets", 0)),
        },
        "metrics": {
            "n_rows_in": float(ds.get("n_rows_in", np.nan)),
            "n_rows_out": float(ds.get("n_rows_out", np.nan)),
            "rows_dropped": float(ds.get("rows_dropped", np.nan)),
            "rows_dropped_ratio": float(ds.get("rows_dropped_ratio", np.nan)),
            "n_days_out": float(ds.get("n_days_out", np.nan)) if ds.get("n_days_out") is not None else np.nan,
        },
    }
    return out


def log_mlflow(summary_mlflow: Dict[str, Any], run_name: str, artifacts: List[str] | None = None, enable: bool = True) -> None:
    if (not enable) or (mlflow is None):
        return

    with mlflow.start_run(run_name=run_name):
        # params
        for k, v in summary_mlflow.get("params", {}).items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        # metrics
        for k, v in summary_mlflow.get("metrics", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        # artifacts
        if artifacts:
            for p in artifacts:
                if p and os.path.exists(p):
                    mlflow.log_artifact(p)


# ============================================================
# 5) Main
# ============================================================
def main() -> None:
    # 1) Load
    log.info("[1] Cargando mnq_intraday_labeled.parquet")
    df_in = load_mnq_parquet(IN_PARQUET)
    df_in = add_column_date(df_in, date_col="date").sort_index()

    # 2) Indicators
    log.info("[2] Calculando indicadores técnicos (por día, sin leakage)")
    df_with_ind, tech_cols = compute_selected_indicators_per_day(df_in, target_col="close", date_col="date")

    # 3) Selección final de columnas
    log.info("[3] Seleccionando features/targets finales")
    ohlc_features = ["open", "high", "low", "close"]  # si quiere OHLCV: agregue "volume"
    final_features = ohlc_features + tech_cols
    final_targets = ["delta_pts_60", "delta_pts_90"]
    selected_columns = ["date"] + final_features + final_targets

    missing_cols = [c for c in selected_columns if c not in df_with_ind.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas para el dataset final: {missing_cols}")

    # 4) Dataset final + dropna
    log.info("[4] Construyendo dataset final y eliminando NaNs")
    df_out = df_with_ind[selected_columns].copy()
    n_before = int(df_out.shape[0])
    df_out = df_out.dropna()
    n_after = int(df_out.shape[0])
    log.info(f"[OK] dropna(): {n_before} -> {n_after} filas")

    # 5) Guardar parquet
    log.info(f"[5] Guardando parquet final: {OUT_PARQUET}")
    _ensure_parent_dir(OUT_PARQUET)
    df_out.to_parquet(OUT_PARQUET, index=False)

    # 6) Summary JSON
    log.info(f"[6] Generando summary: {OUT_SUMMARY}")
    summary = build_stage04_summary_report(
        df_in=df_in,
        df_out=df_out,
        features=final_features,
        targets=final_targets,
        date_col="date",
        decimals=6,
    )
    _ensure_parent_dir(OUT_SUMMARY)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_stage04_summary_pretty(summary)

    # 7) MLflow (opcional)
    mlflow_enable = os.environ.get("MLFLOW_ENABLE", "0") == "1"
    if mlflow_enable:
        log.info("[7] Logging a MLflow")
        mlflow_summary = build_stage04_mlflow_summary(summary)
        log_mlflow(
            mlflow_summary,
            run_name="stage_04_feature_engineering",
            artifacts=[str(OUT_SUMMARY), str(OUT_PARQUET)],
            enable=True,
        )

    log.info("[DONE] stage_04_feature_engineering finalizado correctamente.")


if __name__ == "__main__":
    main()
