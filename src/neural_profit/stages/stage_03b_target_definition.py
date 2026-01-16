"""
stage_03b_target_definition.py

Propósito:
- Definir formalmente los targets de entrenamiento del modelo a partir de los resultados del stage_03a.
- Construir el dataset mnq_intraday_labeled con:
    * delta_pts_h
    * trade_h
    * target_op_h
    * target_tail_h
  para H = 60 y H = 90, evitando data leakage (groupby(date) + shift(-h)).
- Calcular y registrar la "gestation window" (top persistencia |delta_90|/|delta_60|).

Inputs:
- IN_INTRADAY_PARQUET: data/processed/mnq_intraday.parquet
- IN_STAGE03A_SUMMARY: reports/stage_03a_target_investigation_summary.json

Outputs:
- OUT_LABELED_PARQUET: data/processed/mnq_intraday_labeled.parquet

Reports:
- REPORT_SUMMARY: reports/stage_03b_target_definition_summary.json   (envelope estándar)

Params (env/CLI):
- horizons, close_col, date_col
- top_persist_n: cantidad de minutos top para persistencia (ej: 25)
- rr_min (no se usa para labeling, solo referencia si quisieras comparar)
- enable_mlflow (opcional)

Notas:
- Este stage NO modifica datos fuera de:
    - agregar columna date
    - agregar columnas targets
- Gestation window se calcula sobre el dataset labeled (ya con deltas).
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Imports (stdlib)
# ---------------------------------------------------------------------
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Imports (third-party)
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
import logging

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_03b")


# ---------------------------------------------------------------------
# Configuración DVC-friendly (SIEMPRE)
# ---------------------------------------------------------------------
IN_INTRADAY_PARQUET = Path(os.environ.get("IN_INTRADAY_PARQUET", "data/processed/mnq_intraday.parquet"))
IN_STAGE03A_SUMMARY = Path(os.environ.get("IN_STAGE03A_SUMMARY", "reports/stage_03a_target_investigation_summary.json"))

OUT_LABELED_PARQUET = Path(os.environ.get("OUT_LABELED_PARQUET", "data/processed/mnq_intraday_labeled.parquet"))

REPORT_SUMMARY = Path(os.environ.get("REPORT_SUMMARY", "reports/stage_03b_target_definition_summary.json"))

# ---------------------------------------------------------------------
# Params defaults (reproducibles)
# ---------------------------------------------------------------------
STAGE_NAME = "stage_03b_target_definition"
VERSION = "1.0"

DATE_COL = os.environ.get("DATE_COL", "date")
CLOSE_COL = os.environ.get("CLOSE_COL", "close")

HORIZONS = tuple(int(x) for x in os.environ.get("HORIZONS", "60,90").split(","))

DROP_NA_TARGETS = os.environ.get("DROP_NA_TARGETS", "1") in {"1", "true", "True", "yes", "YES"}

# Gestation window / persistencia
TOP_PERSIST_N = int(os.environ.get("TOP_PERSIST_N", "25"))

# MLflow toggle (opcional; no debe romper si mlflow no está instalado)
ENABLE_MLFLOW = os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "yes", "YES"}


# ---------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    _ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary_console(summary: Dict[str, Any]) -> None:
    """Consola legible (sin depender de pandas display)."""
    print("\n" + "=" * 70)
    print(f"STAGE: {summary.get('stage')}")
    print(f"CREATED_AT_UTC: {summary.get('created_at_utc')}")
    print(f"VERSION: {summary.get('version')}")
    print("-" * 70)

    paths = summary.get("paths", {})
    print("[PATHS]")
    for group in ("inputs", "outputs", "reports"):
        print(f"  {group}:")
        for k, v in paths.get(group, {}).items():
            print(f"    - {k}: {v}")

    params = summary.get("params", {})
    print("\n[PARAMS]")
    for k, v in params.items():
        print(f"  - {k}: {v}")

    metrics = summary.get("metrics", {})
    print("\n[METRICS]")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    details = summary.get("details", {})
    gw = details.get("gestation_window", {})
    if gw:
        print("\n[GESTATION WINDOW]")
        print(f"  - start: {gw.get('start_hhmm')}")
        print(f"  - end  : {gw.get('end_hhmm')}")
        print(f"  - top_n: {gw.get('top_n')}")
        print(f"  - minute_of_day_min: {gw.get('minute_of_day_min')}")
        print(f"  - minute_of_day_max: {gw.get('minute_of_day_max')}")

    print("=" * 70 + "\n")


def mlflow_log_from_summary(
    summary: Dict[str, Any],
    *,
    enable: bool,
    run_name: str,
    artifacts: Optional[List[Path]] = None,
) -> None:
    """
    MLflow logging (opcional). No debe romper el pipeline si MLflow no está instalado.
    """
    log.info("[MLFLOW] enable=%s", enable)
    if not enable:
        return

    try:
        import mlflow  # lazy import
    except Exception as exc:
        log.warning("[MLFLOW] mlflow no instalado, omitido. (%s)", exc)
        return

    with mlflow.start_run(run_name=run_name):
        # params
        for k, v in summary.get("params", {}).items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)
            else:
                # asegurar serialización simple
                mlflow.log_param(k, json.dumps(v, ensure_ascii=False))

        # metrics
        for k, v in summary.get("metrics", {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        # tags
        mlflow.set_tag("stage", summary.get("stage", ""))
        mlflow.set_tag("version", summary.get("version", ""))

        # artifacts
        if artifacts:
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))


# ---------------------------------------------------------------------
# Core: carga
# ---------------------------------------------------------------------
def load_intraday_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró IN_INTRADAY_PARQUET: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise TypeError("El intraday parquet debe tener DatetimeIndex o columna 'datetime'.")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def add_column_date(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = out.index.date
    # dejar date primera
    cols = [date_col] + [c for c in out.columns if c != date_col]
    return out[cols]


# ---------------------------------------------------------------------
# Core: cargar artifact stage_03a (ahora es envelope)
# ---------------------------------------------------------------------
def load_stage03a_envelope(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró IN_STAGE03A_SUMMARY: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_deltas_from_stage03a(
    stage03a_env: Dict[str, Any]
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Extrae deltas base/op/tail desde el stage_03a.

    Formato esperado (actual):
      stage03a_env["details"]["summary_rows"] = [
        {"horizon_min": 60, "delta_base_med": ..., "delta_target_p70": ..., "delta_tail_p90": ...},
        {"horizon_min": 90, ...},
      ]

    Fallbacks soportados:
      - stage03a_env["details"]["results"] (si existiera)
      - stage03a_env["results"]
      - formato legacy: stage03a_env es list[dict]
    """

    # 1) localizar la tabla
    results = None

    if isinstance(stage03a_env, list):
        results = stage03a_env
    elif isinstance(stage03a_env, dict):
        details = stage03a_env.get("details", {}) or {}
        results = (
            details.get("summary_rows")
            or details.get("results")
            or stage03a_env.get("results")
        )

    if not isinstance(results, list) or len(results) == 0:
        raise KeyError(
            "No se encontró la tabla de resultados de stage_03a. "
            "Esperado: details.summary_rows (actual) o details.results/results (fallback)."
        )

    # 2) construir mappings
    delta_base_by_h: Dict[int, float] = {}
    delta_op_by_h: Dict[int, float] = {}
    delta_tail_by_h: Dict[int, float] = {}

    for r in results:
        h = int(r["horizon_min"])
        delta_base_by_h[h] = float(r["delta_base_med"])
        delta_op_by_h[h] = float(r["delta_target_p70"])
        delta_tail_by_h[h] = float(r["delta_tail_p90"])

    # 3) sanity check mínimo (por si falta alguno)
    for h in (60, 90):
        if h not in delta_base_by_h or h not in delta_op_by_h or h not in delta_tail_by_h:
            raise KeyError(f"Faltan deltas para h={h} en stage_03a.details.summary_rows.")

    return delta_base_by_h, delta_op_by_h, delta_tail_by_h


# ---------------------------------------------------------------------
# Core: construir labeled dataset (targets)
# ---------------------------------------------------------------------
def make_mnq_intraday_labeled(
    df: pd.DataFrame,
    *,
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

    g = dfx.groupby(date_col, sort=False)

    for h in horizons:
        if h not in delta_base_by_h or h not in delta_op_by_h or h not in delta_tail_by_h:
            raise KeyError(f"Faltan deltas empíricos para h={h}.")

        base_h = float(delta_base_by_h[h])
        op_h = float(delta_op_by_h[h])
        tail_h = float(delta_tail_by_h[h])

        fut_close = g[close_col].shift(-h)
        delta = fut_close - dfx[close_col]

        dfx[f"delta_pts_{h}"] = delta

        trade = np.select([delta >= base_h, delta <= -base_h], [1, -1], default=0).astype("int8")
        dfx[f"trade_{h}"] = trade

        target_op = (((trade == 1) & (delta >= op_h)) | ((trade == -1) & (delta <= -op_h))).astype("int8")
        dfx[f"target_op_{h}"] = target_op

        target_tail = (((trade == 1) & (delta >= tail_h)) | ((trade == -1) & (delta <= -tail_h))).astype("int8")
        dfx[f"target_tail_{h}"] = target_tail

        # invariantes
        flat = (trade == 0)
        dfx.loc[flat, f"target_op_{h}"] = 0
        dfx.loc[flat, f"target_tail_{h}"] = 0

    if drop_na_targets:
        dfx = dfx.dropna(subset=[f"delta_pts_{h}" for h in horizons])

    return dfx


# ---------------------------------------------------------------------
# Core: minute_of_day + métricas persistencia (gestation window)
# ---------------------------------------------------------------------
def add_minute_of_day(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute_of_day"] = out.index.hour * 60 + out.index.minute
    return out


def metrics_by_minute_of_day(
    df: pd.DataFrame,
    thr_pts_by_h: dict,
    trade_60="trade_60",
    trade_90="trade_90",
    d60="delta_pts_60",
    d90="delta_pts_90",
) -> pd.DataFrame:
    dfx = df[[trade_60, trade_90, d60, d90, "minute_of_day"]].copy()

    dfx["abs60"] = dfx[d60].abs()
    dfx["abs90"] = dfx[d90].abs()

    mask_p = (dfx[trade_60] != 0) & (dfx[trade_90] != 0) & (dfx["abs60"] > 0)
    dfx["persist_ratio"] = np.nan
    dfx.loc[mask_p, "persist_ratio"] = dfx.loc[mask_p, "abs90"] / dfx.loc[mask_p, "abs60"]

    def stats_series(x: pd.Series) -> pd.Series:
        x = x.dropna()
        if x.empty:
            return pd.Series({"mean": np.nan, "p50": np.nan, "p75": np.nan, "p90": np.nan})
        return pd.Series({
            "mean": float(x.mean()),
            "p50": float(x.quantile(0.50)),
            "p75": float(x.quantile(0.75)),
            "p90": float(x.quantile(0.90)),
        })

    g = dfx.groupby("minute_of_day", sort=True)

    out = pd.DataFrame(index=g.size().index)

    # ---- Frecuencias 60
    out["n_trade_60"] = (dfx[trade_60] != 0).groupby(dfx["minute_of_day"]).sum().astype(int)
    out["n_long_60"]  = (dfx[trade_60] == 1).groupby(dfx["minute_of_day"]).sum().astype(int)
    out["n_short_60"] = (dfx[trade_60] == -1).groupby(dfx["minute_of_day"]).sum().astype(int)

    # ---- Frecuencias 90
    out["n_trade_90"] = (dfx[trade_90] != 0).groupby(dfx["minute_of_day"]).sum().astype(int)
    out["n_long_90"]  = (dfx[trade_90] == 1).groupby(dfx["minute_of_day"]).sum().astype(int)
    out["n_short_90"] = (dfx[trade_90] == -1).groupby(dfx["minute_of_day"]).sum().astype(int)

    # ---- Stats |delta| condicionados a trade != 0
    abs60_trade = (
        dfx.loc[dfx[trade_60] != 0, ["minute_of_day", "abs60"]]
        .groupby("minute_of_day")["abs60"]
        .apply(stats_series)
        .unstack()
    )
    abs60_trade = abs60_trade.rename(columns={c: f"abs60_trade_{c}" for c in abs60_trade.columns})
    out = out.join(abs60_trade)

    abs90_trade = (
        dfx.loc[dfx[trade_90] != 0, ["minute_of_day", "abs90"]]
        .groupby("minute_of_day")["abs90"]
        .apply(stats_series)
        .unstack()
    )
    abs90_trade = abs90_trade.rename(columns={c: f"abs90_trade_{c}" for c in abs90_trade.columns})
    out = out.join(abs90_trade)

    # ---- Calidad: % trades con |delta| >= thr
    out["pct_abs60_ge_thr_trade"] = (
        (dfx.loc[dfx[trade_60] != 0, "abs60"] >= float(thr_pts_by_h[60]))
        .groupby(dfx.loc[dfx[trade_60] != 0, "minute_of_day"])
        .mean()
    )
    out["pct_abs90_ge_thr_trade"] = (
        (dfx.loc[dfx[trade_90] != 0, "abs90"] >= float(thr_pts_by_h[90]))
        .groupby(dfx.loc[dfx[trade_90] != 0, "minute_of_day"])
        .mean()
    )

    # ---- Persistencia
    out["n_persist"] = dfx["persist_ratio"].notna().groupby(dfx["minute_of_day"]).sum().astype(int)

    persist_stats = (
        dfx[["minute_of_day", "persist_ratio"]]
        .groupby("minute_of_day")["persist_ratio"]
        .apply(stats_series)
        .unstack()
    )
    persist_stats = persist_stats.rename(columns={c: f"persist_ratio_{c}" for c in persist_stats.columns})
    out = out.join(persist_stats)

    # ---- Hour/minute legibles
    out["hour"] = out.index // 60
    out["minute"] = out.index % 60
    out = out[["hour", "minute"] + [c for c in out.columns if c not in ("hour", "minute")]]

    return out

def compute_gestation_window_from_metrics(
    minute_metrics: pd.DataFrame,
    *,
    top_n: int,
) -> Dict[str, Any]:
    """
    Replica tu lógica:
      topPersist = minute_metrics.sort_values("persist_ratio_mean", ascending=False).head(top_n)
      gtn_index_min = topPersist.index.min()
      gtn_index_max = topPersist.index.max()
    """
    if minute_metrics.empty or "persist_ratio_mean" not in minute_metrics.columns:
        return {}

    top_persist = (
        minute_metrics.sort_values("persist_ratio_mean", ascending=False)
        .head(top_n)
        .copy()
    )

    if top_persist.empty:
        return {}

    gtn_index_min = int(top_persist.index.min())
    gtn_index_max = int(top_persist.index.max())

    return {
        "top_n": int(top_n),
        "minute_of_day_min": gtn_index_min,
        "minute_of_day_max": gtn_index_max,
        "start_hhmm": f"{gtn_index_min // 60:02d}:{gtn_index_min % 60:02d}",
        "end_hhmm": f"{gtn_index_max // 60:02d}:{gtn_index_max % 60:02d}",
        # guardamos un extracto mínimo del top para auditoría
        "top_persist_head": top_persist.reset_index()[[
            "minute_of_day", "hour", "minute", "n_persist",
            "persist_ratio_mean", "persist_ratio_p50", "persist_ratio_p90"
        ]].head(min(25, len(top_persist))).to_dict(orient="records")
        if "minute_of_day" in top_persist.reset_index().columns
        else top_persist.reset_index().head(min(25, len(top_persist))).to_dict(orient="records")
    }


# ---------------------------------------------------------------------
# Summary envelope
# ---------------------------------------------------------------------
def build_stage_03b_summary(
    *,
    df_labeled: pd.DataFrame,
    delta_base_by_h: Dict[int, float],
    delta_op_by_h: Dict[int, float],
    delta_tail_by_h: Dict[int, float],
    minute_metrics: pd.DataFrame,
    gestation_window: Dict[str, Any],
    horizons: Tuple[int, ...],
    paths: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:

    n_rows = int(df_labeled.shape[0])
    n_cols = int(df_labeled.shape[1])
    n_days = int(df_labeled[DATE_COL].nunique()) if DATE_COL in df_labeled.columns else 0
    total_nans = int(df_labeled.isna().sum().sum())

    # métricas por horizonte (conteos de targets)
    metrics_by_h: Dict[str, Any] = {}
    for h in horizons:
        trade_col = f"trade_{h}"
        op_col = f"target_op_{h}"
        tail_col = f"target_tail_{h}"
        if trade_col in df_labeled.columns:
            metrics_by_h[f"h{h}_n_trade"] = int((df_labeled[trade_col] != 0).sum())
        if op_col in df_labeled.columns:
            metrics_by_h[f"h{h}_n_target_op"] = int((df_labeled[op_col] == 1).sum())
        if tail_col in df_labeled.columns:
            metrics_by_h[f"h{h}_n_target_tail"] = int((df_labeled[tail_col] == 1).sum())

    summary = {
        "stage": STAGE_NAME,
        "created_at_utc": _iso_utc_now(),
        "version": VERSION,
        "paths": paths,
        "params": params,
        "metrics": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_days": n_days,
            "total_nans": total_nans,
            **metrics_by_h,
        },
        "details": {
            "deltas_used": {
                "delta_base_by_h": {str(k): float(v) for k, v in delta_base_by_h.items()},
                "delta_op_by_h": {str(k): float(v) for k, v in delta_op_by_h.items()},
                "delta_tail_by_h": {str(k): float(v) for k, v in delta_tail_by_h.items()},
            },
            "gestation_window": gestation_window,
            # si querés auditoría liviana, guardamos solo columnas clave del minute_metrics
            "minute_metrics_head": minute_metrics.reset_index().head(15).to_dict(orient="records"),
            "columns": list(df_labeled.columns),
        },
    }
    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="stage_03b_target_definition")
    p.add_argument("--in-intraday-parquet", default=str(IN_INTRADAY_PARQUET))
    p.add_argument("--in-stage03a-summary", default=str(IN_STAGE03A_SUMMARY))
    p.add_argument("--out-labeled-parquet", default=str(OUT_LABELED_PARQUET))
    p.add_argument("--report-summary", default=str(REPORT_SUMMARY))

    p.add_argument("--date-col", default=DATE_COL)
    p.add_argument("--close-col", default=CLOSE_COL)
    p.add_argument("--horizons", default=",".join(str(x) for x in HORIZONS))
    p.add_argument("--drop-na-targets", action="store_true", default=DROP_NA_TARGETS)

    p.add_argument("--top-persist-n", type=int, default=TOP_PERSIST_N)

    p.add_argument("--enable-mlflow", action="store_true", default=ENABLE_MLFLOW)

    return p.parse_args()


# ---------------------------------------------------------------------
# Main (orquestación)
# ---------------------------------------------------------------------
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_intraday = Path(args.in_intraday_parquet)
    in_03a = Path(args.in_stage03a_summary)
    out_labeled = Path(args.out_labeled_parquet)
    report_summary = Path(args.report_summary)

    horizons = tuple(int(x) for x in str(args.horizons).split(","))

    paths = {
        "inputs": {
            "intraday_parquet": str(in_intraday.as_posix()),
            "stage03a_summary": str(in_03a.as_posix()),
        },
        "outputs": {
            "mnq_intraday_labeled": str(out_labeled.as_posix()),
        },
        "reports": {
            "summary": str(report_summary.as_posix()),
        },
    }

    params = {
        "date_col": args.date_col,
        "close_col": args.close_col,
        "horizons": list(horizons),
        "drop_na_targets": bool(args.drop_na_targets),
        "top_persist_n": int(args.top_persist_n),
    }

    log.info("[1] Cargando intraday parquet: %s", in_intraday)
    df = load_intraday_parquet(in_intraday)
    df = add_column_date(df, date_col=args.date_col)

    log.info("[2] Cargando stage_03a summary envelope: %s", in_03a)
    s03a = load_stage03a_envelope(in_03a)

    log.info("[3] Extrayendo deltas (base/op/tail) desde stage_03a")
    delta_base_by_h, delta_op_by_h, delta_tail_by_h = extract_deltas_from_stage03a(s03a)

    log.info("[4] Construyendo dataset labeled (no leakage por día)")
    labeled = make_mnq_intraday_labeled(
        df,
        date_col=args.date_col,
        close_col=args.close_col,
        horizons=horizons,
        delta_base_by_h=delta_base_by_h,
        delta_op_by_h=delta_op_by_h,
        delta_tail_by_h=delta_tail_by_h,
        drop_na_targets=bool(args.drop_na_targets),
    )

    log.info("[5] Agregando minute_of_day y calculando persistencia")
    labeled = add_minute_of_day(labeled)

    thr_pts_by_h = {
        60: float(delta_base_by_h[60]),
        90: float(delta_base_by_h[90]),
    }

    minute_metrics = metrics_by_minute_of_day(labeled, thr_pts_by_h=thr_pts_by_h)
    gestation_window = compute_gestation_window_from_metrics(minute_metrics, top_n=int(args.top_persist_n))

    log.info(
        "[5.1] Gestation window: %s -> %s",
        gestation_window.get("start_hhmm"),
        gestation_window.get("end_hhmm"),
    )

    log.info("[6] Guardando parquet labeled: %s", out_labeled)
    _ensure_parent_dir(out_labeled)
    labeled.to_parquet(out_labeled, index=True)

    log.info("[7] Construyendo y guardando summary envelope: %s", report_summary)
    summary = build_stage_03b_summary(
        df_labeled=labeled,
        delta_base_by_h=delta_base_by_h,
        delta_op_by_h=delta_op_by_h,
        delta_tail_by_h=delta_tail_by_h,
        minute_metrics=minute_metrics,
        gestation_window=gestation_window,
        horizons=horizons,
        paths=paths,
        params=params,
    )
    save_json(summary, report_summary)

    log.info("[8] MLflow tracking (enable=%s)", bool(args.enable_mlflow))
    mlflow_log_from_summary(
        summary,
        enable=bool(args.enable_mlflow),
        run_name=STAGE_NAME,
        artifacts=[report_summary],
    )

    print_summary_console(summary)
    log.info("[OK] stage_03b completado")


if __name__ == "__main__":
    main()
