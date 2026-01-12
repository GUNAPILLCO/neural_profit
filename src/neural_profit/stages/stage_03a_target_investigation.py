# ============================================================
# stage_03a_target_investigation.py
#
# Objetivo:
# - Analizar empíricamente movimientos intradía (Δpts) por minuto del día
# - Derivar deltas operativos (base / p70 / p90) para H=60 y H=90
# - Estimar Stop Loss empírico (MAE hasta TP=delta_target_p70)
# - Construir una tabla resumen final con RR y stop recomendado
#
# Entradas / salidas (vía env vars):
#   IN_PARQUET  : data/processed/mnq_intraday.parquet
#   OUT_SUMMARY : reports/target_investigation_summary.json
# ============================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from numba import njit

import logging 

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("stage_03a")

#MLflow
try:
    import mlflow
except ImportError:
    mlflow = None


# ----------------------------
# IO (paths por env vars)
# ----------------------------
IN_PARQUET = Path(os.environ.get("IN_PARQUET", "data/processed/mnq_intraday.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/target_investigation_summary.json"))


# ============================================================
# 1) Carga y preprocesamiento base
# ============================================================
def load_mnq_parquet(path: Path = IN_PARQUET) -> pd.DataFrame:
    """Carga el parquet intradía."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")
    log.info(f"[OK] Archivo encontrado. Cargando: {path}")
    return pd.read_parquet(path)


def add_column_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura índice datetime y crea columna 'date' (fecha sin hora).
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out["date"] = out.index.date
    cols = ["date"] + [c for c in out.columns if c != "date"]
    return out[cols]


def add_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega:
      - minute_of_day = hour*60 + minute
      - hour, minute
    """
    out = df.copy()
    out["minute_of_day"] = out.index.hour * 60 + out.index.minute
    out["hour"] = out.index.hour
    out["minute"] = out.index.minute
    return out


# ============================================================
# 2) Cálculo de deltas futuros por día (sin cruzar sesión)
# ============================================================
def add_future_delta_pts_by_day(
    df: pd.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    horizons: Tuple[int, ...] = (60, 90),
) -> pd.DataFrame:
    """
    Por cada horizonte h:
      Δpts_{t,h} = close_{t+h} - close_t   (shift(-h) dentro del mismo día)
      abs_delta_pts_{h} = |Δ|
      sign_delta_{h} = sign(Δ)
    """
    out = df.copy()
    g = out.groupby(date_col, sort=False)

    for h in horizons:
        fut_close = g[close_col].shift(-h)
        out[f"delta_pts_{h}"] = fut_close - out[close_col]
        out[f"abs_delta_pts_{h}"] = out[f"delta_pts_{h}"].abs()
        out[f"sign_delta_{h}"] = np.sign(out[f"delta_pts_{h}"]).astype("float")

    return out


# ============================================================
# 3) Estadísticos por minuto del día (ranking)
# ============================================================
def stats_by_minute_of_day(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Estadísticos por minute_of_day para un horizonte h:
      - n
      - mean_abs, median_abs
      - percentiles p60, p70, p80, p90 de |Δ|
      - pos_ratio / neg_ratio (sesgo direccional simple)
    """
    base = df[["minute_of_day", f"delta_pts_{h}", f"abs_delta_pts_{h}"]].dropna()
    g = base.groupby("minute_of_day", sort=True)

    out = g.agg(
        n=(f"abs_delta_pts_{h}", "size"),
        mean_abs=(f"abs_delta_pts_{h}", "mean"),
        median_abs=(f"abs_delta_pts_{h}", "median"),
    )

    for q in [0.60, 0.70, 0.80, 0.90]:
        out[f"p{int(q*100)}_abs"] = g[f"abs_delta_pts_{h}"].quantile(q)

    out["pos_ratio"] = g[f"delta_pts_{h}"].apply(lambda x: float((x > 0).mean()))
    out["neg_ratio"] = g[f"delta_pts_{h}"].apply(lambda x: float((x < 0).mean()))

    out["hour"] = (out.index // 60).astype(int)
    out["minute"] = (out.index % 60).astype(int)

    return out.sort_index()


def top_minutes(df_by_time: pd.DataFrame, col: str, top_n: int = 30) -> pd.DataFrame:
    """
    Top-N minutos del día según 'col'. Imprime ventana horaria del top.
    """
    cols_show = [
        "hour", "minute", "n",
        "mean_abs", "median_abs",
        "p60_abs", "p70_abs", "p80_abs", "p90_abs",
        "pos_ratio", "neg_ratio",
    ]

    top_df = (
        df_by_time.sort_values(col, ascending=False)[cols_show]
        .head(top_n)
        .copy()
    )

    # Ventana (min/max) dentro del top
    minute_of_day = top_df["hour"] * 60 + top_df["minute"]
    min_mod = int(minute_of_day.min())
    max_mod = int(minute_of_day.max())

    min_time = f"{min_mod//60:02d}:{min_mod%60:02d}"
    max_time = f"{max_mod//60:02d}:{max_mod%60:02d}"

    min_val = float(top_df[col].min())
    max_val = float(top_df[col].max())

    #print("=" * 60)
    #print(f"Top {top_n} según '{col}'")
    #print(f"Ventana horaria: {min_time}  ->  {max_time}")
    #print(f"{col} mínimo: {min_val:.2f}")
    #print(f"{col} máximo: {max_val:.2f}")
    #print("=" * 60)

    return top_df


def delta_midpoint(delta_min: float, delta_max: float) -> float:
    """Punto medio entre min y max (robusto como estimador operativo)."""
    return float(delta_min + (delta_max - delta_min) / 2.0)


def optimal_window_from_rankings(*tops: pd.DataFrame) -> Dict[str, Any]:
    """
    Intersección horaria robusta a partir de varios rankings (top_med/top_p70/top_p90).
    """
    starts, ends = [], []
    for df in tops:
        mod = df["hour"] * 60 + df["minute"]
        starts.append(int(mod.min()))
        ends.append(int(mod.max()))

    start_opt = max(starts)
    end_opt = min(ends)
    if start_opt > end_opt:
        raise ValueError("No hay intersección horaria entre rankings.")

    return {
        "start_minute_of_day": start_opt,
        "end_minute_of_day": end_opt,
        "start_hhmm": f"{start_opt // 60:02d}:{start_opt % 60:02d}",
        "end_hhmm": f"{end_opt // 60:02d}:{end_opt % 60:02d}",
    }


# ============================================================
# 4) Stop Loss empírico (MAE hasta TP)
# ============================================================
@njit
def _mae_until_tp_numba(close, high, low, sign, h, tp_pts):
    """
    MAE hasta TP o hasta horizonte h (lo que ocurra primero), por día.
    Devuelve:
      mae[i]   : adverse excursion en pts
      tp_hit[i]: True si tocó TP dentro de h
      tau[i]   : minutos hasta salida (TP o vencimiento)
    """
    n = close.shape[0]
    mae = np.full(n, np.nan)
    tp_hit = np.zeros(n, dtype=np.bool_)
    tau = np.full(n, np.nan)

    for i in range(n):
        if i + h >= n:
            continue

        s = sign[i]
        if not np.isfinite(s) or s == 0.0:
            continue

        entry = close[i]
        u_exit = h  # default: vence horizonte

        if s > 0:  # LONG
            tp_price = entry + tp_pts

            # primer u donde high >= tp_price
            for u in range(1, h + 1):
                if high[i + u] >= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

            # MAE LONG: entry - min(low) en [i+1, i+u_exit]
            min_low = low[i + 1]
            for u in range(2, u_exit + 1):
                v = low[i + u]
                if v < min_low:
                    min_low = v

            mae[i] = entry - min_low
            tau[i] = u_exit

        else:      # SHORT
            tp_price = entry - tp_pts

            # primer u donde low <= tp_price
            for u in range(1, h + 1):
                if low[i + u] <= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

            # MAE SHORT: max(high) en [i+1, i+u_exit] - entry
            max_high = high[i + 1]
            for u in range(2, u_exit + 1):
                v = high[i + u]
                if v > max_high:
                    max_high = v

            mae[i] = max_high - entry
            tau[i] = u_exit

    return mae, tp_hit, tau


def compute_mae_until_tp_fast(
    df: pd.DataFrame,
    h: int,
    tp_pts: float,
    date_col: str = "date",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    sign_col: str | None = None,
) -> pd.DataFrame:
    """
    Wrapper por día + Numba.
    Retorna columnas:
      mae_tp_{h}, tp_hit_{h}, tau_{h}
    """
    if sign_col is None:
        sign_col = f"sign_delta_{h}"

    required = {date_col, close_col, high_col, low_col, sign_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    mae_out = np.full(len(df), np.nan, dtype=float)
    hit_out = np.zeros(len(df), dtype=bool)
    tau_out = np.full(len(df), np.nan, dtype=float)

    # Mantener orden del DF: asignación por posiciones
    for _, g in df.groupby(date_col, sort=False):
        idx = g.index.to_numpy()
        locs = df.index.get_indexer(idx)

        close = g[close_col].to_numpy(dtype=np.float64)
        high = g[high_col].to_numpy(dtype=np.float64)
        low  = g[low_col].to_numpy(dtype=np.float64)
        sign = g[sign_col].to_numpy(dtype=np.float64)

        mae_d, hit_d, tau_d = _mae_until_tp_numba(close, high, low, sign, h, tp_pts)

        mae_out[locs] = mae_d
        hit_out[locs] = hit_d
        tau_out[locs] = tau_d

    return pd.DataFrame(
        {f"mae_tp_{h}": mae_out, f"tp_hit_{h}": hit_out, f"tau_{h}": tau_out},
        index=df.index,
    )


def mae_percentiles_by_time(
    df: pd.DataFrame,
    mae_col: str,
    percentiles=(0.7, 0.8),
    minute_col: str = "minute_of_day",
) -> pd.DataFrame:
    """Percentiles del MAE por minute_of_day."""
    base = df[[minute_col, mae_col]].dropna().copy()
    g = base.groupby(minute_col)[mae_col]

    out = pd.DataFrame(index=g.size().index)
    out["n"] = g.size()

    for p in percentiles:
        out[f"p{int(p*100)}"] = g.quantile(p)

    out = out.reset_index()
    out["hour"] = (out[minute_col] // 60).astype(int)
    out["minute"] = (out[minute_col] % 60).astype(int)
    out["time_hm"] = out["hour"].astype(str).str.zfill(2) + ":" + out["minute"].astype(str).str.zfill(2)

    return out.sort_values(minute_col).reset_index(drop=True)


def top_minutes_mae(mae_pct: pd.DataFrame, col: str, top_n: int = 30) -> pd.DataFrame:
    """Top-N minutos según percentil MAE (p70/p80)."""
    cols_show = ["hour", "minute", "time_hm", "n", col]
    return mae_pct.sort_values(col, ascending=False)[cols_show].head(top_n).copy()


def filter_window(df: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    """Filtra por ventana HH:MM (incluye ambos extremos)."""
    start = int(start_hhmm.split(":")[0]) * 60 + int(start_hhmm.split(":")[1])
    end = int(end_hhmm.split(":")[0]) * 60 + int(end_hhmm.split(":")[1])
    return df[(df["minute_of_day"] >= start) & (df["minute_of_day"] <= end)].copy()


# ============================================================
# 5) Construcción del summary report
# ============================================================
def build_stage03a_summary_report(
    delta_base_med_60, delta_target_p70_60, delta_tail_p90_60,
    delta_base_med_90, delta_target_p70_90, delta_tail_p90_90,
    window_60: Dict[str, Any],
    window_90: Dict[str, Any],
    stop_loss_60_p70, stop_loss_60_p80,
    stop_loss_90_p70, stop_loss_90_p80,
    rr_min: float = 2.0,
    decimals: int = 2
) -> pd.DataFrame:
    """
    Construye el summary técnico del stage 03a.
    No imprime ni formatea para consola.
    """
    rows = [
        {
            "horizon_min": 60,
            "optimal_window": f"{window_60['start_hhmm']}-{window_60['end_hhmm']}",
            "delta_base_med": delta_base_med_60,
            "delta_target_p70": delta_target_p70_60,
            "delta_tail_p90": delta_tail_p90_60,
            "stop_loss_p70": stop_loss_60_p70,
            "stop_loss_p80": stop_loss_60_p80,
        },
        {
            "horizon_min": 90,
            "optimal_window": f"{window_90['start_hhmm']}-{window_90['end_hhmm']}",
            "delta_base_med": delta_base_med_90,
            "delta_target_p70": delta_target_p70_90,
            "delta_tail_p90": delta_tail_p90_90,
            "stop_loss_p70": stop_loss_90_p70,
            "stop_loss_p80": stop_loss_90_p80,
        },
    ]

    df = pd.DataFrame(rows)

    # Reward / Risk
    df["RR_target_p70_vs_SL_p70"] = df["delta_target_p70"] / df["stop_loss_p70"]
    df["RR_target_p70_vs_SL_p80"] = df["delta_target_p70"] / df["stop_loss_p80"]
    df["RR_tail_p90_vs_SL_p70"]   = df["delta_tail_p90"]   / df["stop_loss_p70"]
    df["RR_tail_p90_vs_SL_p80"]   = df["delta_tail_p90"]   / df["stop_loss_p80"]

    def recommend_stop(row):
        rr_p70 = row["RR_target_p70_vs_SL_p70"]
        if pd.notna(rr_p70) and rr_p70 >= rr_min:
            return pd.Series({
                "stop_recomendado_tipo": "p70",
                "stop_recomendado": row["stop_loss_p70"],
                "RR_recomendado": rr_p70,
            })
        return pd.Series({
            "stop_recomendado_tipo": "p80",
            "stop_recomendado": row["stop_loss_p80"],
            "RR_recomendado": row["RR_target_p70_vs_SL_p80"],
        })

    df = pd.concat([df, df.apply(recommend_stop, axis=1)], axis=1)

    # Redondeo técnico
    for c in df.columns:
        if c not in ["horizon_min", "optimal_window", "stop_recomendado_tipo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].round(decimals)

    return df

# ============================================================
# Visualización
# ============================================================


def print_stage03a_summary_pretty(df: pd.DataFrame) -> None:
    """
    Presentación amigable para consola (Git Bash).
    """
    def f(x):
        return "-" if pd.isna(x) else f"{float(x):.2f}"

    print("\nObjetivos operativos dictados por el mercado:\n")

    for _, r in df.iterrows():
        h = int(r["horizon_min"])
        print("=" * 70)
        print(f"RESUMEN H={h} min   |   Ventana óptima: {r['optimal_window']}")
        print("-" * 70)
        print(
            f"Deltas (pts)        | "
            f"base: {f(r['delta_base_med'])}   "
            f"target: {f(r['delta_target_p70'])}   "
            f"tail: {f(r['delta_tail_p90'])}"
        )
        print(
            f"Stop Loss (pts)     | "
            f"p70: {f(r['stop_loss_p70'])}   "
            f"p80: {f(r['stop_loss_p80'])}   "
            f"recomendado: {r['stop_recomendado_tipo']} -> {f(r['stop_recomendado'])}"
        )
        print(
            f"Risk/Reward         | "
            f"TP/SL(p70): {f(r['RR_target_p70_vs_SL_p70'])}   "
            f"TP/SL(p80): {f(r['RR_target_p70_vs_SL_p80'])}"
        )
        print(
            f"                   | "
            f"Tail/SL(p70): {f(r['RR_tail_p90_vs_SL_p70'])}   "
            f"Tail/SL(p80): {f(r['RR_tail_p90_vs_SL_p80'])}"
        )
        print(f"RR recomendado      | {f(r['RR_recomendado'])}")

    print("=" * 70)


# ============================================================
# 6) MLFlow
# ============================================================

# ============================================================
# 6.1) Construcción de mlflow_summary
# ============================================================
def build_stage03a_mlflow_summary(
    summary_df: pd.DataFrame,
    rr_min: float
) -> dict:
    """
    Summary reducido (params + metrics) para MLflow.
    """
    out = {
        "params": {
            "rr_min": rr_min,
            "horizons": summary_df["horizon_min"].tolist(),
        },
        "metrics": {}
    }

    for _, r in summary_df.iterrows():
        h = int(r["horizon_min"])
        out["metrics"].update({
            f"h{h}_delta_base": float(r["delta_base_med"]),
            f"h{h}_delta_target_p70": float(r["delta_target_p70"]),
            f"h{h}_delta_tail_p90": float(r["delta_tail_p90"]),
            f"h{h}_stop_recommended": float(r["stop_recomendado"]),
            f"h{h}_RR_recommended": float(r["RR_recomendado"]),
        })

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
    # ---- 6.1 Carga ----
    log.info("[1] Cargando dataset mnq_intraday.parquet")
    mnq = load_mnq_parquet(IN_PARQUET)
    mnq = add_column_date(mnq)
    mnq = mnq.sort_index()

    # ---- 6.2 Deltas + minuto del día ----
    log.info("[2] Agregando columna 'date' y asegurando orden temporal")
    mnq = add_future_delta_pts_by_day(mnq, horizons=(60, 90))
    mnq = add_minute_of_day(mnq)

    # ---- 6.3 Stats por minuto (ranking) ----
    log.info("[3] Calculando deltas futuros (H=60, H=90)")
    by_time_60 = stats_by_minute_of_day(mnq, h=60)
    by_time_90 = stats_by_minute_of_day(mnq, h=90)

    # Rankings top-30 por métrica (mediana/p70/p90)
    log.info("[4] Construyendo rankings por minuto")
    top_med_60 = top_minutes(by_time_60, col="median_abs", top_n=30)
    top_p70_60 = top_minutes(by_time_60, col="p70_abs", top_n=30)
    top_p90_60 = top_minutes(by_time_60, col="p90_abs", top_n=30)

    top_med_90 = top_minutes(by_time_90, col="median_abs", top_n=30)
    top_p70_90 = top_minutes(by_time_90, col="p70_abs", top_n=30)
    top_p90_90 = top_minutes(by_time_90, col="p90_abs", top_n=30)

    # ---- 6.4 Deltas operativos (punto medio del rango top) ----
    log.info("[5] Calculando los deltas operativos (median /p70 / p90)")
    delta_base_med_60 = delta_midpoint(top_med_60["median_abs"].min(), top_med_60["median_abs"].max())
    delta_target_p70_60 = delta_midpoint(top_p70_60["p70_abs"].min(), top_p70_60["p70_abs"].max())
    delta_tail_p90_60 = delta_midpoint(top_p90_60["p90_abs"].min(), top_p90_60["p90_abs"].max())

    delta_base_med_90 = delta_midpoint(top_med_90["median_abs"].min(), top_med_90["median_abs"].max())
    delta_target_p70_90 = delta_midpoint(top_p70_90["p70_abs"].min(), top_p70_90["p70_abs"].max())
    delta_tail_p90_90 = delta_midpoint(top_p90_90["p90_abs"].min(), top_p90_90["p90_abs"].max())

    # ---- 6.5 Ventanas óptimas (intersección robusta) ----
    log.info("[6] Determinación de ventanas de óptimas de operación")
    optimal_window_60 = optimal_window_from_rankings(top_med_60, top_p70_60, top_p90_60)
    optimal_window_90 = optimal_window_from_rankings(top_med_90, top_p70_90, top_p90_90)

    # ---- 6.6 MAE hasta TP (tp_pts = delta_target_p70_h) ----
    log.info("[7] Calculando MAE hasta TakeProfit (Stop Loss empírico)")
    mae_60 = compute_mae_until_tp_fast(mnq, h=60, tp_pts=float(delta_target_p70_60))
    mae_90 = compute_mae_until_tp_fast(mnq, h=90, tp_pts=float(delta_target_p70_90))
    mnq_mae = mnq.join(mae_60).join(mae_90)

    # Percentiles de MAE por minuto del día
    mae_pct_60 = mae_percentiles_by_time(mnq_mae, mae_col="mae_tp_60", percentiles=(0.7, 0.8))
    mae_pct_90 = mae_percentiles_by_time(mnq_mae, mae_col="mae_tp_90", percentiles=(0.7, 0.8))

    # ---- 6.7 Stop Loss por ventana óptima (mediana dentro de la ventana) ----
    # Filtramos percentiles MAE dentro de la ventana óptima del movimiento (no la de MAE)
    log.info("[8] Buscando los mejores Stop Loss para ventanas de operación")
    mae_pct_60_win = filter_window(mae_pct_60, optimal_window_60["start_hhmm"], optimal_window_60["end_hhmm"])
    mae_pct_90_win = filter_window(mae_pct_90, optimal_window_90["start_hhmm"], optimal_window_90["end_hhmm"])

    stop_loss_60_p70 = float(mae_pct_60_win["p70"].median())
    stop_loss_60_p80 = float(mae_pct_60_win["p80"].median())
    stop_loss_90_p70 = float(mae_pct_90_win["p70"].median())
    stop_loss_90_p80 = float(mae_pct_90_win["p80"].median())

    # ---- 6.8 Tabla final (RR + stop recomendado) ----
    log.info("[9] Construyendo target_investigation_summary.json")
    summary_report_stage_03a = build_stage03a_summary_report(
        delta_base_med_60, delta_target_p70_60, delta_tail_p90_60,
        delta_base_med_90, delta_target_p70_90, delta_tail_p90_90,
        window_60=optimal_window_60,
        window_90=optimal_window_90,
        stop_loss_60_p70=stop_loss_60_p70,
        stop_loss_60_p80=stop_loss_60_p80,
        stop_loss_90_p70=stop_loss_90_p70,
        stop_loss_90_p80=stop_loss_90_p80,
        decimals=2
    )
    
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    summary_report_stage_03a.to_json(
        OUT_SUMMARY, 
        orient="records", 
        indent=2)

    log.info("[OK] Summary report escrito en: %s", OUT_SUMMARY)
    

    log.info("[10] Construyendo stage_03a_mlflow_summary")
    mlflow_summary = build_stage03a_mlflow_summary(
        summary_report_stage_03a,
        rr_min=2.0
    )

    log.info("[11] Logging MLflow")
    log_mlflow(
        summary=mlflow_summary,
        run_name="stage_03a_target_investigation",
        artifacts=[str(OUT_SUMMARY)],
        enable=bool(int(os.environ.get("ENABLE_MLFLOW", "0")))
    )

    print_stage03a_summary_pretty(summary_report_stage_03a)


if __name__ == "__main__":
    main()
