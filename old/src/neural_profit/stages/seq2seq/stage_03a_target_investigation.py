"""
stage_03a_target_investigation

Propósito
- Analizar movimientos intradía (Δpts) por minuto del día.
- Derivar deltas operativos (base / target_p70 / tail_p90) para H=60 y H=90.
- Estimar Stop Loss empírico (MAE hasta TP=delta_target_p70).
- Producir UN SOLO JSON con envelope estándar + rows legacy dentro.

Inputs (DVC deps)
- IN_INTRADAY_PARQUET: data/processed/mnq_intraday.parquet

Reports
- REPORT_SUMMARY: reports/stage_03a_target_investigation_summary.json
  (envelope estándar, incluye details.summary_rows con la estructura legacy)

Notas
- No mezcla días: shifts y MAE se calculan por 'date'.
- No entrena modelos.
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
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------------------------------------------------
# Imports (third-party)
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
from numba import njit

# ---------------------------------------------------------------------
# Logging (uniforme)
# ---------------------------------------------------------------------
import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_03a")

# ---------------------------------------------------------------------
# Configuración DVC-friendly (SIEMPRE presente)
# ---------------------------------------------------------------------
IN_INTRADAY_PARQUET = Path(os.environ.get("IN_INTRADAY_PARQUET", "data/processed/mnq_intraday.parquet"))
REPORT_SUMMARY = Path(os.environ.get("REPORT_SUMMARY", "reports/stage_03a_target_investigation_summary.json"))

# ---------------------------------------------------------------------
# Configuración funcional (parámetros del stage)
# ---------------------------------------------------------------------
HORIZONS: Tuple[int, ...] = (60, 90)
TOP_N = int(os.environ.get("TOP_N", "30"))
RR_MIN = float(os.environ.get("RR_MIN", "2.0"))
DECIMALS = int(os.environ.get("DECIMALS", "2"))

# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(payload: Any, path: Path) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Core: carga y columnas base
# ---------------------------------------------------------------------
def load_mnq_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise TypeError("El DataFrame debe tener DatetimeIndex o columna 'datetime'.")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def add_column_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out.index.date
    cols = ["date"] + [c for c in out.columns if c != "date"]
    return out[cols]


def add_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute_of_day"] = out.index.hour * 60 + out.index.minute
    out["hour"] = out.index.hour
    out["minute"] = out.index.minute
    return out


def add_future_delta_pts_by_day(
    df: pd.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    horizons: Tuple[int, ...] = (60, 90),
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(date_col, sort=False)

    for h in horizons:
        fut_close = g[close_col].shift(-h)
        out[f"delta_pts_{h}"] = fut_close - out[close_col]
        out[f"abs_delta_pts_{h}"] = out[f"delta_pts_{h}"].abs()
        out[f"sign_delta_{h}"] = np.sign(out[f"delta_pts_{h}"]).astype("float")

    return out


# ---------------------------------------------------------------------
# Stats por minuto del día
# ---------------------------------------------------------------------
def stats_by_minute_of_day(df: pd.DataFrame, h: int) -> pd.DataFrame:
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
    cols_show = [
        "hour", "minute", "n",
        "mean_abs", "median_abs",
        "p60_abs", "p70_abs", "p80_abs", "p90_abs",
        "pos_ratio", "neg_ratio",
    ]
    return df_by_time.sort_values(col, ascending=False)[cols_show].head(top_n).copy()


def delta_midpoint(delta_min: float, delta_max: float) -> float:
    return float(delta_min + (delta_max - delta_min) / 2.0)


def optimal_window_from_rankings(*tops: pd.DataFrame) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------
# MAE hasta TP (numba)
# ---------------------------------------------------------------------
@njit
def _mae_until_tp_numba(close, high, low, sign, h, tp_pts):
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
        u_exit = h

        if s > 0:  # LONG
            tp_price = entry + tp_pts
            for u in range(1, h + 1):
                if high[i + u] >= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

            min_low = low[i + 1]
            for u in range(2, u_exit + 1):
                v = low[i + u]
                if v < min_low:
                    min_low = v

            mae[i] = entry - min_low
            tau[i] = u_exit

        else:  # SHORT
            tp_price = entry - tp_pts
            for u in range(1, h + 1):
                if low[i + u] <= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

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
    if sign_col is None:
        sign_col = f"sign_delta_{h}"

    required = {date_col, close_col, high_col, low_col, sign_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    mae_out = np.full(len(df), np.nan, dtype=float)
    hit_out = np.zeros(len(df), dtype=bool)
    tau_out = np.full(len(df), np.nan, dtype=float)

    for _, g in df.groupby(date_col, sort=False):
        idx = g.index.to_numpy()
        locs = df.index.get_indexer(idx)

        close = g[close_col].to_numpy(dtype=np.float64)
        high = g[high_col].to_numpy(dtype=np.float64)
        low = g[low_col].to_numpy(dtype=np.float64)
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
    base = df[[minute_col, mae_col]].dropna().copy()
    g = base.groupby(minute_col)[mae_col]

    out = pd.DataFrame(index=g.size().index)
    out["n"] = g.size()

    for p in percentiles:
        out[f"p{int(p*100)}"] = g.quantile(p)

    out = out.reset_index()
    out["hour"] = (out[minute_col] // 60).astype(int)
    out["minute"] = (out[minute_col] % 60).astype(int)
    return out.sort_values(minute_col).reset_index(drop=True)


def filter_window(mae_pct: pd.DataFrame, start_hhmm: str, end_hhmm: str, minute_col: str = "minute_of_day") -> pd.DataFrame:
    start = int(start_hhmm.split(":")[0]) * 60 + int(start_hhmm.split(":")[1])
    end = int(end_hhmm.split(":")[0]) * 60 + int(end_hhmm.split(":")[1])
    return mae_pct[(mae_pct[minute_col] >= start) & (mae_pct[minute_col] <= end)].copy()


# ---------------------------------------------------------------------
# Summary legacy (tabla final) y envelope
# ---------------------------------------------------------------------
def build_summary_table(
    *,
    delta_base_med_60: float,
    delta_target_p70_60: float,
    delta_tail_p90_60: float,
    delta_base_med_90: float,
    delta_target_p70_90: float,
    delta_tail_p90_90: float,
    window_60: Dict[str, Any],
    window_90: Dict[str, Any],
    stop_loss_60_p70: float,
    stop_loss_60_p80: float,
    stop_loss_90_p70: float,
    stop_loss_90_p80: float,
    rr_min: float,
    decimals: int,
) -> pd.DataFrame:
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

    df["RR_target_p70_vs_SL_p70"] = df["delta_target_p70"] / df["stop_loss_p70"]
    df["RR_target_p70_vs_SL_p80"] = df["delta_target_p70"] / df["stop_loss_p80"]
    df["RR_tail_p90_vs_SL_p70"] = df["delta_tail_p90"] / df["stop_loss_p70"]
    df["RR_tail_p90_vs_SL_p80"] = df["delta_tail_p90"] / df["stop_loss_p80"]

    def recommend_stop(row: pd.Series) -> pd.Series:
        rr_p70 = row["RR_target_p70_vs_SL_p70"]
        if pd.notna(rr_p70) and rr_p70 >= rr_min:
            return pd.Series(
                {"stop_recomendado_tipo": "p70", "stop_recomendado": row["stop_loss_p70"], "RR_recomendado": rr_p70}
            )
        return pd.Series(
            {"stop_recomendado_tipo": "p80", "stop_recomendado": row["stop_loss_p80"], "RR_recomendado": row["RR_target_p70_vs_SL_p80"]}
        )

    df = pd.concat([df, df.apply(recommend_stop, axis=1)], axis=1)

    for c in df.columns:
        if c not in ["horizon_min", "optimal_window", "stop_recomendado_tipo"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].round(decimals)

    return df


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def build_envelope(
    *,
    in_path: Path,
    report_path: Path,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    summary_rows: List[Dict[str, Any]],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": "stage_03a_target_investigation",
        "created_at_utc": _iso_utc_now(),
        "version": "1.0",
        "paths": {
            "inputs": {"intraday_parquet": str(in_path.as_posix())},
            "outputs": {},
            "reports": {"summary": str(report_path.as_posix())},
        },
        "params": params,
        "metrics": metrics,
        "details": {
            # <- AQUÍ está tu estructura legacy, intacta, en un solo archivo
            "summary_rows": summary_rows,
            **details,
        },
    }


def print_stage_03a_summary_console(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("STAGE: stage_03a_target_investigation")
    print("=" * 70)
    for r in summary_rows:
        h = r.get("horizon_min")
        print("-" * 70)
        print(f"H={h} | ventana={r.get('optimal_window')}")
        print(f"  delta_base_med     : {r.get('delta_base_med')}")
        print(f"  delta_target_p70   : {r.get('delta_target_p70')}")
        print(f"  delta_tail_p90     : {r.get('delta_tail_p90')}")
        print(f"  stop_loss_p70      : {r.get('stop_loss_p70')}")
        print(f"  stop_loss_p80      : {r.get('stop_loss_p80')}")
        print(f"  stop recomendado   : {r.get('stop_recomendado_tipo')} -> {r.get('stop_recomendado')}")
        print(f"  RR recomendado     : {r.get('RR_recomendado')}")
    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------
# MLflow (opcional, estándar)
# ---------------------------------------------------------------------
def mlflow_log_from_envelope(envelope: Dict[str, Any], *, enable: bool) -> None:
    if not enable:
        log.info("[MLFLOW] enable=False -> omitido")
        return

    try:
        import mlflow
    except Exception as exc:
        log.warning("[MLFLOW] No disponible (pip install mlflow). Omitiendo. Detalle: %s", exc)
        return

    run_name = envelope.get("stage", "stage_03a_target_investigation")
    params = envelope.get("params", {}) or {}
    metrics = envelope.get("metrics", {}) or {}
    report_path = (envelope.get("paths", {}) or {}).get("reports", {}).get("summary", "")

    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            try:
                if isinstance(v, (dict, list)):
                    mlflow.log_param(k, json.dumps(v, ensure_ascii=False))
                else:
                    mlflow.log_param(k, v)
            except Exception:
                pass

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        try:
            if report_path and os.path.exists(report_path):
                mlflow.log_artifact(report_path)
        except Exception:
            pass


# ---------------------------------------------------------------------
# parse_args (override estándar + alias legacy)
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="stage_03a_target_investigation (MNQ)")

    p.add_argument("--in-intraday-parquet", default=str(IN_INTRADAY_PARQUET))
    p.add_argument("--report-summary", default=str(REPORT_SUMMARY))

    p.add_argument("--top-n", type=int, default=TOP_N)
    p.add_argument("--rr-min", type=float, default=RR_MIN)
    p.add_argument("--decimals", type=int, default=DECIMALS)

    p.add_argument(
        "--enable-mlflow",
        action="store_true",
        default=os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "YES", "yes"},
    )

    # aliases (por si algún dvc.yaml viejo usa estos)
    p.add_argument("--in-parquet", dest="in_intraday_parquet", help=argparse.SUPPRESS)
    p.add_argument("--out-summary", dest="report_summary", help=argparse.SUPPRESS)

    return p.parse_args()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_parquet = Path(args.in_intraday_parquet)
    report_summary = Path(args.report_summary)

    log.info("[1] Cargando intraday parquet: %s", in_parquet)
    df = load_mnq_parquet(in_parquet)

    log.info("[2] Agregando columnas base y asegurando orden temporal")
    df = add_column_date(df).sort_index()
    df = add_future_delta_pts_by_day(df, horizons=HORIZONS)
    df = add_minute_of_day(df)

    log.info("[3] Stats por minute_of_day (H=60/90)")
    by_time = {h: stats_by_minute_of_day(df, h=h) for h in HORIZONS}

    log.info("[4] Rankings top-%s por median/p70/p90", args.top_n)
    top = {}
    for h in HORIZONS:
        top[(h, "median_abs")] = top_minutes(by_time[h], col="median_abs", top_n=args.top_n)
        top[(h, "p70_abs")] = top_minutes(by_time[h], col="p70_abs", top_n=args.top_n)
        top[(h, "p90_abs")] = top_minutes(by_time[h], col="p90_abs", top_n=args.top_n)

    log.info("[5] Deltas operativos (midpoint del rango top)")
    d_base_60 = delta_midpoint(top[(60, "median_abs")]["median_abs"].min(), top[(60, "median_abs")]["median_abs"].max())
    d_tgt_60 = delta_midpoint(top[(60, "p70_abs")]["p70_abs"].min(), top[(60, "p70_abs")]["p70_abs"].max())
    d_tail_60 = delta_midpoint(top[(60, "p90_abs")]["p90_abs"].min(), top[(60, "p90_abs")]["p90_abs"].max())

    d_base_90 = delta_midpoint(top[(90, "median_abs")]["median_abs"].min(), top[(90, "median_abs")]["median_abs"].max())
    d_tgt_90 = delta_midpoint(top[(90, "p70_abs")]["p70_abs"].min(), top[(90, "p70_abs")]["p70_abs"].max())
    d_tail_90 = delta_midpoint(top[(90, "p90_abs")]["p90_abs"].min(), top[(90, "p90_abs")]["p90_abs"].max())

    log.info("[6] Ventanas óptimas (intersección)")
    w60 = optimal_window_from_rankings(top[(60, "median_abs")], top[(60, "p70_abs")], top[(60, "p90_abs")])
    w90 = optimal_window_from_rankings(top[(90, "median_abs")], top[(90, "p70_abs")], top[(90, "p90_abs")])

    log.info("[7] MAE hasta TP (tp_pts = delta_target_p70)")
    mae_60 = compute_mae_until_tp_fast(df, h=60, tp_pts=float(d_tgt_60))
    mae_90 = compute_mae_until_tp_fast(df, h=90, tp_pts=float(d_tgt_90))
    df_mae = df.join(mae_60).join(mae_90)

    mae_pct_60 = mae_percentiles_by_time(df_mae, mae_col="mae_tp_60", percentiles=(0.7, 0.8))
    mae_pct_90 = mae_percentiles_by_time(df_mae, mae_col="mae_tp_90", percentiles=(0.7, 0.8))

    log.info("[8] Stop loss por ventana óptima (mediana dentro de la ventana)")
    mae_pct_60_win = filter_window(mae_pct_60, w60["start_hhmm"], w60["end_hhmm"])
    mae_pct_90_win = filter_window(mae_pct_90, w90["start_hhmm"], w90["end_hhmm"])

    sl60_p70 = float(mae_pct_60_win["p70"].median())
    sl60_p80 = float(mae_pct_60_win["p80"].median())
    sl90_p70 = float(mae_pct_90_win["p70"].median())
    sl90_p80 = float(mae_pct_90_win["p80"].median())

    log.info("[9] Construyendo summary rows (legacy) + envelope único")
    summary_df = build_summary_table(
        delta_base_med_60=d_base_60,
        delta_target_p70_60=d_tgt_60,
        delta_tail_p90_60=d_tail_60,
        delta_base_med_90=d_base_90,
        delta_target_p70_90=d_tgt_90,
        delta_tail_p90_90=d_tail_90,
        window_60=w60,
        window_90=w90,
        stop_loss_60_p70=sl60_p70,
        stop_loss_60_p80=sl60_p80,
        stop_loss_90_p70=sl90_p70,
        stop_loss_90_p80=sl90_p80,
        rr_min=float(args.rr_min),
        decimals=int(args.decimals),
    )

    summary_rows = df_to_records(summary_df)  # <- mantiene el formato que me pediste

    params = {
        "horizons": list(HORIZONS),
        "top_n": int(args.top_n),
        "rr_min": float(args.rr_min),
        "decimals": int(args.decimals),
    }

    metrics = {
        "h60_delta_base_med": float(summary_df.loc[summary_df["horizon_min"] == 60, "delta_base_med"].iloc[0]),
        "h60_delta_target_p70": float(summary_df.loc[summary_df["horizon_min"] == 60, "delta_target_p70"].iloc[0]),
        "h60_delta_tail_p90": float(summary_df.loc[summary_df["horizon_min"] == 60, "delta_tail_p90"].iloc[0]),
        "h60_stop_recommended": float(summary_df.loc[summary_df["horizon_min"] == 60, "stop_recomendado"].iloc[0]),
        "h60_rr_recommended": float(summary_df.loc[summary_df["horizon_min"] == 60, "RR_recomendado"].iloc[0]),
        "h90_delta_base_med": float(summary_df.loc[summary_df["horizon_min"] == 90, "delta_base_med"].iloc[0]),
        "h90_delta_target_p70": float(summary_df.loc[summary_df["horizon_min"] == 90, "delta_target_p70"].iloc[0]),
        "h90_delta_tail_p90": float(summary_df.loc[summary_df["horizon_min"] == 90, "delta_tail_p90"].iloc[0]),
        "h90_stop_recommended": float(summary_df.loc[summary_df["horizon_min"] == 90, "stop_recomendado"].iloc[0]),
        "h90_rr_recommended": float(summary_df.loc[summary_df["horizon_min"] == 90, "RR_recomendado"].iloc[0]),
    }

    details = {
        "optimal_window_60": w60,
        "optimal_window_90": w90,
        "stop_loss_window_medians": {
            "h60": {"p70": sl60_p70, "p80": sl60_p80},
            "h90": {"p70": sl90_p70, "p80": sl90_p80},
        },
    }

    envelope = build_envelope(
        in_path=in_parquet,
        report_path=report_summary,
        params=params,
        metrics=metrics,
        summary_rows=summary_rows,
        details=details,
    )

    log.info("[10] Guardando JSON único (envelope + legacy rows): %s", report_summary)
    _save_json(envelope, report_summary)

    log.info("[11] MLflow tracking (enable=%s)", args.enable_mlflow)
    mlflow_log_from_envelope(envelope, enable=bool(args.enable_mlflow))

    print_stage_03a_summary_console(summary_rows)

    log.info("[OK] Stage_03a completo. Report: %s", report_summary)


if __name__ == "__main__":
    main()
