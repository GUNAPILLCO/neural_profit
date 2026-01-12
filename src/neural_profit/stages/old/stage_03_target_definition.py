# ============================================================
# stage_03_target_definition (ejecutable .py)
# Targets: trade_h (dir + base) y hold_h (extensión condicional)
# Horizontes: (60, 90)
# ============================================================

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Iterable, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
@dataclass
class Stage03Config:
    # IO
    input_path: str = "data/processed/mnq_intraday.parquet"
    output_labeled_path: str = "data/processed/mnq_intraday_labeled.parquet"
    summary_path: str = "reports/target_definition_summary.json"

    # Columns
    datetime_col: Optional[str] = None  # si None, se usa el índice si es DatetimeIndex
    date_col: str = "date"
    close_col: str = "close"

    # Targets
    horizons: Tuple[int, ...] = (60, 90)

    # Umbrales económicos (en puntos)
    base_pts: float = 25.0     # Δ_base
    ext_pts: float = 62.5      # Δ_ext

    # Output behavior
    drop_na_targets: bool = False
    timezone_expected: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _as_dtindex(df: pd.DataFrame, datetime_col: Optional[str]) -> pd.DataFrame:
    dfx = df.copy()
    if datetime_col is None:
        if not isinstance(dfx.index, pd.DatetimeIndex):
            raise TypeError("Se esperaba df.index como DatetimeIndex. O indique datetime_col en config.")
        return dfx.sort_index()

    if datetime_col not in dfx.columns:
        raise KeyError(f"datetime_col='{datetime_col}' no existe en el DataFrame.")

    dfx[datetime_col] = pd.to_datetime(dfx[datetime_col], utc=False, errors="raise")
    dfx = dfx.set_index(datetime_col).sort_index()

    if not isinstance(dfx.index, pd.DatetimeIndex):
        raise TypeError("No se pudo convertir datetime_col a DatetimeIndex.")
    return dfx


def _infer_or_build_date_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dfx = df.copy()
    if date_col not in dfx.columns:
        dfx[date_col] = pd.to_datetime(dfx.index.date)
    else:
        dfx[date_col] = pd.to_datetime(dfx[date_col])
    return dfx


def _basic_target_stats(x: pd.Series) -> Dict[str, Any]:
    x = x.dropna()
    if x.empty:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "max": None,
        }
    return {
        "count": int(x.shape[0]),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "min": float(x.min()),
        "p01": float(x.quantile(0.01)),
        "p05": float(x.quantile(0.05)),
        "p50": float(x.quantile(0.50)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def _pct_nans(x: pd.Series) -> float:
    return float(x.isna().mean())


def _safe_tz_name(df: pd.DataFrame) -> Optional[str]:
    try:
        return str(df.index.tz) if df.index.tz is not None else None
    except Exception:
        return None


# ----------------------------
# Core: Target Definition
# ----------------------------
def make_mnq_intraday_labeled(
    df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "close",
    horizons: Iterable[int] = (60, 90),
    base_pts: float = 25.0,
    ext_pts: float = 62.5,
    drop_na_targets: bool = False,
) -> pd.DataFrame:
    """
    Genera targets por día sin cruzar sesiones.

    Δpts_{t,h} = close_{t+h} - close_t    (shift(-h) dentro del día)

    trade_h:
      +1 si Δpts >=  base_pts
      -1 si Δpts <= -base_pts
       0 si |Δpts| < base_pts

    hold_h (condicionada a trade):
      1 si trade_h = +1 y Δpts >=  ext_pts
      1 si trade_h = -1 y Δpts <= -ext_pts
      0 en caso contrario

    Restricción: trade_h = 0 => hold_h = 0 (se cumple por construcción).
    """
    dfx = df.copy()

    if close_col not in dfx.columns:
        raise KeyError(f"Falta close_col='{close_col}'.")
    if date_col not in dfx.columns:
        raise KeyError(f"Falta date_col='{date_col}' (para no cruzar sesiones).")

    dfx = dfx.sort_index()

    for h in horizons:
        fut_close = dfx.groupby(date_col, sort=False)[close_col].shift(-h)
        delta = fut_close - dfx[close_col]
        dfx[f"delta_pts_{h}"] = delta

        trade = np.where(delta >= base_pts, 1, np.where(delta <= -base_pts, -1, 0)).astype("int8")
        dfx[f"trade_{h}"] = trade

        hold = np.zeros(len(dfx), dtype="int8")
        hold[(trade == 1) & (delta >= ext_pts)] = 1
        hold[(trade == -1) & (delta <= -ext_pts)] = 1
        dfx[f"hold_{h}"] = hold

    if drop_na_targets:
        dfx = dfx.dropna(subset=[f"delta_pts_{h}" for h in horizons])

    return dfx


# ----------------------------
# Summary (JSON report)
# ----------------------------
def build_target_definition_summary(df_labeled: pd.DataFrame, cfg: Stage03Config) -> Dict[str, Any]:
    horizons = cfg.horizons

    tzname = _safe_tz_name(df_labeled)
    dt_min = df_labeled.index.min()
    dt_max = df_labeled.index.max()
    n_rows, n_cols = map(int, df_labeled.shape)
    n_days = int(df_labeled[cfg.date_col].nunique()) if cfg.date_col in df_labeled.columns else None

    minutes_per_day = (
        df_labeled.groupby(cfg.date_col).size().describe().to_dict()
        if cfg.date_col in df_labeled.columns else None
    )

    targets: Dict[str, Any] = {}
    for h in horizons:
        key = f"h{h}"
        delta_col = f"delta_pts_{h}"
        trade_col = f"trade_{h}"
        hold_col = f"hold_{h}"

        pct_nan_delta = _pct_nans(df_labeled[delta_col])
        delta_stats = _basic_target_stats(df_labeled[delta_col])

        trade_counts = df_labeled[trade_col].value_counts(dropna=False).to_dict()
        hold_rate = float((df_labeled[hold_col] == 1).mean())

        trade_mask = df_labeled[trade_col] != 0
        hold_rate_given_trade = float(df_labeled.loc[trade_mask, hold_col].mean()) if trade_mask.any() else None

        delta_on_trade = df_labeled.loc[trade_mask, delta_col]
        abs_delta_on_trade = delta_on_trade.abs()

        targets[key] = {
            "horizon_min": int(h),
            "definition": {
                "delta_pts": "close_{t+h} - close_t",
                "trade": f"+1 if delta>= {cfg.base_pts}, -1 if delta<= -{cfg.base_pts}, else 0",
                "hold": (
                    f"1 if (trade=+1 and delta>= {cfg.ext_pts}) or (trade=-1 and delta<= -{cfg.ext_pts}); else 0. "
                    "Also trade=0 => hold=0."
                ),
            },
            "columns": {"delta_pts": delta_col, "trade": trade_col, "hold": hold_col},
            "pct_nans_delta_pts": pct_nan_delta,
            "delta_pts_stats": delta_stats,
            "trade_class_counts": {str(k): int(v) for k, v in trade_counts.items()},
            "hold_rate": hold_rate,
            "hold_rate_given_trade": hold_rate_given_trade,
            "delta_pts_on_trade_stats": _basic_target_stats(delta_on_trade),
            "abs_delta_pts_on_trade_stats": _basic_target_stats(abs_delta_on_trade),
        }

    summary = {
        "stage": "stage_03_target_definition",
        "description": (
            "Definición explícita de variables objetivo (trade/hold) para horizontes 60/90. "
            "Cálculo por día sin cruzar sesiones: groupby(date) + shift(-h)."
        ),
        "input": cfg.input_path,
        "output_labeled": cfg.output_labeled_path,
        "artifacts": [cfg.summary_path],
        "params": {
            "horizons": list(cfg.horizons),
            "base_pts": cfg.base_pts,
            "ext_pts": cfg.ext_pts,
            "drop_na_targets": cfg.drop_na_targets,
            "date_col": cfg.date_col,
            "close_col": cfg.close_col,
        },
        "dataset_info": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "datetime_min": str(dt_min),
            "datetime_max": str(dt_max),
            "timezone": tzname,
            "n_days": n_days,
            "minutes_per_day_describe": minutes_per_day,
            "final_target_columns": [
                *[f"delta_pts_{h}" for h in horizons],
                *[f"trade_{h}" for h in horizons],
                *[f"hold_{h}" for h in horizons],
            ],
        },
        "targets": targets,
        "minimum_metrics": {
            "pct_nans_per_target": {k: v["pct_nans_delta_pts"] for k, v in targets.items()},
            "basic_stats_per_target": {k: v["delta_pts_stats"] for k, v in targets.items()},
        },
    }
    return summary


# ----------------------------
# Runner
# ----------------------------
def run_stage_03(cfg: Stage03Config) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not os.path.exists(cfg.input_path):
        raise FileNotFoundError(f"Input no encontrado: {cfg.input_path}")

    df_raw = pd.read_parquet(cfg.input_path)
    df_raw = _as_dtindex(df_raw, cfg.datetime_col)

    if cfg.timezone_expected is not None:
        tzname = _safe_tz_name(df_raw)
        if tzname != cfg.timezone_expected:
            print(f"[WARN] timezone={tzname} (esperado={cfg.timezone_expected}).")

    df_raw = _infer_or_build_date_col(df_raw, cfg.date_col)

    df_labeled = make_mnq_intraday_labeled(
        df_raw,
        date_col=cfg.date_col,
        close_col=cfg.close_col,
        horizons=cfg.horizons,
        base_pts=cfg.base_pts,
        ext_pts=cfg.ext_pts,
        drop_na_targets=cfg.drop_na_targets,
    )

    _ensure_parent_dir(cfg.output_labeled_path)
    df_labeled.to_parquet(cfg.output_labeled_path, index=True)

    summary = build_target_definition_summary(df_labeled, cfg)
    _ensure_parent_dir(cfg.summary_path)
    with open(cfg.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return df_labeled, summary


# ----------------------------
# CLI
# ----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="stage_03_target_definition: genera mnq_intraday_labeled + summary JSON")

    p.add_argument("--input-path", default="data/processed/mnq_intraday.parquet")
    p.add_argument("--output-labeled-path", default="data/processed/mnq_intraday_labeled.parquet")
    p.add_argument("--summary-path", default="reports/target_definition_summary.json")

    p.add_argument("--datetime-col", default=None, help="Si no se usa índice DatetimeIndex, indique la columna datetime")
    p.add_argument("--date-col", default="date")
    p.add_argument("--close-col", default="close")

    p.add_argument("--horizons", nargs="+", type=int, default=[60, 90], help="Ej: --horizons 60 90")
    p.add_argument("--base-pts", type=float, default=25.0)
    p.add_argument("--ext-pts", type=float, default=62.5)

    p.add_argument("--drop-na-targets", action="store_true", help="Si se activa, elimina filas con delta_pts_h NaN")
    p.add_argument("--timezone-expected", default=None, help="Nombre tz esperado (opcional), ej: America/New_York")

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg = Stage03Config(
        input_path=args.input_path,
        output_labeled_path=args.output_labeled_path,
        summary_path=args.summary_path,
        datetime_col=args.datetime_col,
        date_col=args.date_col,
        close_col=args.close_col,
        horizons=tuple(args.horizons),
        base_pts=args.base_pts,
        ext_pts=args.ext_pts,
        drop_na_targets=bool(args.drop_na_targets),
        timezone_expected=args.timezone_expected,
    )

    print("[stage_03] Config:")
    print(json.dumps({
        "input_path": cfg.input_path,
        "output_labeled_path": cfg.output_labeled_path,
        "summary_path": cfg.summary_path,
        "datetime_col": cfg.datetime_col,
        "date_col": cfg.date_col,
        "close_col": cfg.close_col,
        "horizons": list(cfg.horizons),
        "base_pts": cfg.base_pts,
        "ext_pts": cfg.ext_pts,
        "drop_na_targets": cfg.drop_na_targets,
        "timezone_expected": cfg.timezone_expected,
    }, indent=2, ensure_ascii=False))

    try:
        df_labeled, summary = run_stage_03(cfg)
    except Exception as e:
        print(f"[stage_03][ERROR] {type(e).__name__}: {e}")
        return 1

    print("[stage_03] OK")
    print(f"[stage_03] output: {cfg.output_labeled_path}  rows={len(df_labeled)} cols={df_labeled.shape[1]}")
    print(f"[stage_03] summary: {cfg.summary_path}")
    # Métricas mínimas rápidas
    for h in cfg.horizons:
        print(
            f"[stage_03] h={h}  pct_nan(delta)={summary['targets'][f'h{h}']['pct_nans_delta_pts']:.4f}  "
            f"hold_rate={summary['targets'][f'h{h}']['hold_rate']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
