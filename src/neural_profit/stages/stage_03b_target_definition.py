# ============================================================
# stage_03b_target_definition.py
#
# Objetivo:
# - Definir formalmente los targets de entrenamiento del modelo
#   a partir de los resultados empíricos del stage_03a
# - Construir el dataset mnq_intraday_labeled con:
#     * delta_pts_h        : movimiento futuro en puntos
#     * trade_h            : señal direccional (umbral base)
#     * target_op_h        : objetivo operativo (p70)
#     * target_tail_h      : objetivo de extensión / cola (p90)
#   para horizontes H = 60 y H = 90
# - Garantizar separación temporal estricta (no data leakage)
#   mediante cálculo por día: groupby(date) + shift(-h)
# - Generar un summary liviano del dataset final
#   (dimensiones, columnas, NaNs, horarios y parámetros usados)
#
# Entradas / salidas (vía env vars):
#   IN_PARQUET   : data/processed/mnq_intraday.parquet
#   IN_ARTIFACT  : reports/target_investigation_summary.json
#   OUT_PARQUET  : data/processed/mnq_intraday_labeled.parquet
#   OUT_SUMMARY  : reports/target_definition_summary.json
#
# Dependencias:
# - Requiere haber ejecutado previamente stage_03a_target_investigation
#   para obtener los deltas empíricos (base / p70 / p90)
# ============================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("stage_03b")

#MLflow
try:
    import mlflow
except ImportError:
    mlflow = None


# ----------------------------
# IO (paths por env vars)
# ----------------------------
IN_PARQUET  = Path(os.environ.get("IN_PARQUET",  "data/processed/mnq_intraday.parquet"))
IN_ARTIFACT = Path(os.environ.get("IN_ARTIFACT", "reports/target_investigation_summary.json"))
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET", "data/processed/mnq_intraday_labeled.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/target_definition_summary.json"))

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

def add_column_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out["date"] = out.index.date
    cols = ["date"] + [c for c in out.columns if c != "date"]
    return out[cols]

# ============================================================
# 2) Carga artifact stage_03a
# ============================================================
def load_artifact_stage_03a(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el artifact de 03a: {path}")
    log.info(f"[OK] Cargando artifact 03a: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 3) Dataset labeled (targets)
# ============================================================
def make_mnq_intraday_labeled(
    df: pd.DataFrame,
    date_col: str,
    close_col: str,
    horizons: Iterable[int],
    delta_base_by_h: Dict[int, float],
    delta_op_by_h: Dict[int, float],
    delta_tail_by_h: Dict[int, float],
    drop_na_targets: bool = True,
) -> pd.DataFrame:

    dfx = df.copy()

    if close_col not in dfx.columns:
        raise KeyError(f"Falta close_col='{close_col}'.")
    if date_col not in dfx.columns:
        raise KeyError(f"Falta date_col='{date_col}' (para no cruzar sesiones).")

    dfx = dfx.sort_index()

    for h in horizons:
        if h not in delta_base_by_h or h not in delta_op_by_h or h not in delta_tail_by_h:
            raise KeyError(f"Faltan deltas empíricos para h={h}.")

        base_h = float(delta_base_by_h[h])
        op_h   = float(delta_op_by_h[h])
        tail_h = float(delta_tail_by_h[h])

        fut_close = dfx.groupby(date_col, sort=False)[close_col].shift(-h)
        delta = fut_close - dfx[close_col]

        dfx[f"delta_pts_{h}"] = delta

        trade = np.select([delta >= base_h, delta <= -base_h], [1, -1], default=0).astype("int8")
        dfx[f"trade_{h}"] = trade

        target_op = (((trade == 1) & (delta >= op_h)) | ((trade == -1) & (delta <= -op_h))).astype("int8")
        dfx[f"target_op_{h}"] = target_op

        target_tail = (((trade == 1) & (delta >= tail_h)) | ((trade == -1) & (delta <= -tail_h))).astype("int8")
        dfx[f"target_tail_{h}"] = target_tail

        # Asegurar restricción: trade=0 => targets=0
        flat = (trade == 0)
        dfx.loc[flat, f"target_op_{h}"] = 0
        dfx.loc[flat, f"target_tail_{h}"] = 0

    if drop_na_targets:
        dfx = dfx.dropna(subset=[f"delta_pts_{h}" for h in horizons])

    return dfx

# ============================================================
# 4) Summary liviano (JSON)
# ============================================================
def build_stage03b_summary_report(
    df_labeled: pd.DataFrame,
    delta_base_by_h: Dict[int, float],
    delta_op_by_h: Dict[int, float],
    delta_tail_by_h: Dict[int, float],
    horizons=(60, 90),
    date_col: str = "date",
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Construye el summary tabular del stage 03b.
    Devuelve un DataFrame (1 fila por horizonte), alineado con stage_03a.
    """

    n_rows = int(df_labeled.shape[0])
    n_days = int(df_labeled[date_col].nunique())
    rows_per_day_mean = df_labeled.groupby(date_col).size().mean()

    rows = []
    for h in horizons:
        rows.append({
            "horizon_min": h,
            "delta_base_pts": delta_base_by_h[h],
            "delta_op_pts": delta_op_by_h[h],
            "delta_tail_pts": delta_tail_by_h[h],
            "target_columns": f"trade_{h}, target_op_{h}, target_tail_{h}",
            "n_rows": n_rows,
            "n_days": n_days,
            "rows_per_day_mean": rows_per_day_mean,
        })

    df = pd.DataFrame(rows)

    # Redondeo técnico
    for c in df.columns:
        if c not in ["horizon_min", "target_columns"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].round(decimals)

    return df



def print_stage03b_summary_pretty(df: pd.DataFrame) -> None:
    """
    Imprime un resumen legible en consola del stage 03b
    a partir del DataFrame summary (una fila por horizonte).
    """

    if df.empty:
        print("Summary vacío (stage 03b).")
        return

    print("\n" + "=" * 78)
    print("DEFINICIÓN FINAL DE TARGETS – DATASET MNQ (Stage 03b)")
    print("=" * 78)

    def f2(x):
        return "-" if pd.isna(x) else f"{float(x):.2f}"

    for _, r in df.iterrows():
        h = int(r["horizon_min"])

        print("\n" + "-" * 78)
        print(f"Horizonte H = {h} min")
        print("-" * 78)

        print(
            f"Deltas (pts) | "
            f"base: {f2(r['delta_base_pts'])}   "
            f"op (p70): {f2(r['delta_op_pts'])}   "
            f"tail (p90): {f2(r['delta_tail_pts'])}"
        )

        print(
            f"Targets     | "
            f"trade_{h}, target_op_{h}, target_tail_{h}"
        )

    print("\n" + "=" * 78)

# ============================================================
# 6) MLFlow
# ============================================================

# ============================================================
# 6.1) Construcción de mlflow_summary
# ============================================================
def build_stage03b_mlflow_summary(
    summary_df: pd.DataFrame,
    horizons: tuple[int, ...] = (60, 90),
) -> dict:
    """
    Summary reducido (params + metrics) para MLflow (Stage 03b).

    Espera un DataFrame tipo "summary report" (una fila por horizonte),
    con columnas como:
      - horizon_min
      - delta_base_med, delta_target_p70, delta_tail_p90
      - (opcional) optimal_window, stop_recomendado, RR_recomendado, etc.

    Nota: Stage 03b define targets; por eso lo más importante a trackear
    en MLflow son los umbrales (deltas) por horizonte.
    """
    out = {
        "params": {
            "horizons": list(horizons),
            "target_schema": "trade_h + target_op_h + target_tail_h",
        },
        "metrics": {}
    }

    if summary_df is None or summary_df.empty:
        return out

    for _, r in summary_df.iterrows():
        h = int(r["horizon_min"])

        # Deltas usados para construir el dataset etiquetado
        out["metrics"].update({
            #f"h{h}_delta_base": float(r["delta_base_med"]),
            #f"h{h}_delta_op_p70": float(r["delta_target_p70"]),
            #f"h{h}_delta_tail_p90": float(r["delta_tail_p90"]),
            f"h{h}_delta_base": float(r["delta_base_pts"]),
            f"h{h}_delta_op_p70": float(r["delta_op_pts"]),
            f"h{h}_delta_tail_p90": float(r["delta_tail_pts"]),
        })

        # Opcionales (si existen en el DF)
        if "stop_recomendado" in summary_df.columns and pd.notna(r.get("stop_recomendado", np.nan)):
            out["metrics"][f"h{h}_stop_recommended"] = float(r["stop_recomendado"])

        if "RR_recomendado" in summary_df.columns and pd.notna(r.get("RR_recomendado", np.nan)):
            out["metrics"][f"h{h}_RR_recommended"] = float(r["RR_recomendado"])

    return out
# ============================================================
# 6.2) Logging 
# ============================================================

def log_mlflow(summary: dict, run_name: str, artifacts: list[str] = None, enable: bool = True):
    if (not enable) or (mlflow is None):
        return

    with mlflow.start_run(run_name=run_name):
        # params (planos y chicos)
        for k, v in summary.get("params", {}).items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        # metrics (solo numéricas)
        def _log_metrics(prefix, d):
            for k, v in d.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"{prefix}{k}", float(v))

        _log_metrics("", summary.get("metrics", {}))

        # artifacts
        if artifacts:
            for p in artifacts:
                if p and os.path.exists(p):
                    mlflow.log_artifact(p)



# ============================================================
# 7) Main
# ============================================================
def main() -> None:
    log.info("[1] Cargando mnq_intraday.parquet")
    mnq = load_mnq_parquet(IN_PARQUET)
    mnq = add_column_date(mnq).sort_index()

    log.info("[2] Cargando artifact de 03a")
    a3 = load_artifact_stage_03a(IN_ARTIFACT)

    log.info("[3] Extrayendo deltas (base/op/tail)")
    #delta_base_by_h = {60: a3["horizons"]["60"]["deltas"]["base_med"],   90: a3["horizons"]["90"]["deltas"]["base_med"]}
    #delta_op_by_h   = {60: a3["horizons"]["60"]["deltas"]["target_p70"], 90: a3["horizons"]["90"]["deltas"]["target_p70"]}
    #delta_tail_by_h = {60: a3["horizons"]["60"]["deltas"]["tail_p90"],   90: a3["horizons"]["90"]["deltas"]["tail_p90"]}
    
    delta_base_by_h = {60: a3[0]["delta_base_med"], 90: a3[1]["delta_base_med"]}
    delta_op_by_h = {60: a3[0]["delta_target_p70"], 90: a3[1]["delta_target_p70"]}
    delta_tail_by_h = {60: a3[0]["delta_tail_p90"], 90: a3[1]["delta_tail_p90"]}
    

    log.info("[4] Generando mnq_intraday_labeled")
    labeled = make_mnq_intraday_labeled(
        mnq,
        date_col="date",
        close_col="close",
        horizons=(60, 90),
        delta_base_by_h=delta_base_by_h,
        delta_op_by_h=delta_op_by_h,
        delta_tail_by_h=delta_tail_by_h,
        drop_na_targets=True,
    )

    log.info("[5] Guardando outputs")
    _ensure_parent_dir(OUT_PARQUET)
    labeled.to_parquet(OUT_PARQUET, index=True)

    log.info("[6] Construyendo target_definition_summary.json")

    summary_report_stage_03b = build_stage03b_summary_report(
    df_labeled=labeled,
    #artifact_03a=a3,
    delta_base_by_h=delta_base_by_h,
    delta_op_by_h=delta_op_by_h,
    delta_tail_by_h=delta_tail_by_h,
    horizons=(60, 90),
    date_col="date",
    ) 

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    summary_report_stage_03b.to_json(
        OUT_SUMMARY, 
        orient="records", 
        indent=2)

    log.info("[OK] Summary report escrito en: %s", OUT_SUMMARY)    
    
    log.info("[10] Construyendo stage_03b_mlflow_summary")
    mlflow_summary = build_stage03b_mlflow_summary(
        summary_report_stage_03b,
        horizons=(60, 90)
    )

    log.info("[11] Logging MLflow")
    log_mlflow(
        summary=mlflow_summary,
        run_name="stage_03b_target_investigation",
        artifacts=[str(OUT_SUMMARY)],
        enable=bool(int(os.environ.get("ENABLE_MLFLOW", "0")))
    )
        
    
    print_stage03b_summary_pretty(summary_report_stage_03b)


if __name__ == "__main__":
    main()

