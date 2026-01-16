"""
stage_04_feature_engineering.py

Stage: stage_04_feature_engineering
Propósito:
- Construir features intradía (OHLC + indicadores técnicos) a partir de mnq_intraday_labeled
- Mantener cálculo "time-safe": indicadores calculados por día (sin cruzar jornadas)
- Generar dataset final features+targets y un summary envelope para auditoría/MLflow

Inputs (deps):
- IN_LABELED_PARQUET: data/processed/mnq_intraday_labeled.parquet

Outputs (outs):
- OUT_FEATURES_PARQUET: data/features/mnq_features_target.parquet

Reports (reports):
- REPORT_SUMMARY: reports/stage_04_feature_engineering_summary.json

Params (defaults reproducibles):
- PARAM_USE_VOLUME
- PARAM_DROPNA_POLICY
- PARAM_INDICATORS: EMA span, momentum lag, ROC windows

Notas:
- No genera targets; asume que delta_pts_{60,90} ya existen en el labeled.
- Dropna final aplica a columnas finales seleccionadas (date + features + targets).
"""

from __future__ import annotations

# ============================================================
# 1) Imports
# ============================================================
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator

# ============================================================
# 2) Logging (uniforme)
# ============================================================
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_04")

# ============================================================
# 3) Configuración DVC-friendly (SIEMPRE presente)
# ============================================================
IN_LABELED_PARQUET = Path(os.environ.get("IN_LABELED_PARQUET", "data/processed/mnq_intraday_labeled.parquet"))

OUT_FEATURES_PARQUET = Path(os.environ.get("OUT_FEATURES_PARQUET", "data/features/mnq_features_target.parquet"))

REPORT_SUMMARY = Path(os.environ.get("REPORT_SUMMARY", "reports/stage_04_feature_engineering_summary.json"))

# ============================================================
# 4) Configuración funcional (params del stage)
# ============================================================
PARAM_DATE_COL = os.environ.get("PARAM_DATE_COL", "date")
PARAM_CLOSE_COL = os.environ.get("PARAM_CLOSE_COL", "close")

PARAM_USE_VOLUME = os.environ.get("PARAM_USE_VOLUME", "0") in {"1", "true", "True", "YES", "yes"}

# Indicadores
PARAM_EMA_SPAN = int(os.environ.get("PARAM_EMA_SPAN", "60"))
PARAM_MOMENTUM_LAG = int(os.environ.get("PARAM_MOMENTUM_LAG", "10"))
PARAM_ROC_30 = int(os.environ.get("PARAM_ROC_30", "30"))
PARAM_ROC_60 = int(os.environ.get("PARAM_ROC_60", "60"))

# Targets esperados en labeled
PARAM_TARGETS = tuple(int(x) for x in os.environ.get("PARAM_TARGETS", "60,90").split(","))

# Drop policy: "any" (dropna sobre todas las cols finales), "targets" (dropna solo en targets)
PARAM_DROPNA_POLICY = os.environ.get("PARAM_DROPNA_POLICY", "any").strip().lower()

# MLflow (opcional)
ENABLE_MLFLOW = os.environ.get("ENABLE_MLFLOW", "0") in {"1", "true", "True", "YES", "yes"}


# ============================================================
# 5) Utilidades puras
# ============================================================
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_parquet_as_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.set_index("datetime")
        else:
            raise TypeError("El DataFrame debe tener DatetimeIndex o una columna 'datetime'.")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def add_or_replace_date_col(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out[date_col] = out.index.date
    # dejar date primero
    cols = [date_col] + [c for c in out.columns if c != date_col]
    return out[cols]


def compute_indicators_per_day(
    df: pd.DataFrame,
    *,
    date_col: str,
    close_col: str,
    ema_span: int,
    momentum_lag: int,
    roc_30: int,
    roc_60: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calcula indicadores de forma independiente por día (sin leakage inter-día).
    Devuelve df_out (mismas filas) + lista de columnas creadas.
    """
    required = {date_col, close_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    indicator_cols = ["price_ema60", "momentum_10", "roc_30", "roc_60"]

    def per_day(day_df: pd.DataFrame) -> pd.DataFrame:
        d = day_df.copy()

        # EMA normalized extension
        ema = d[close_col].ewm(span=ema_span, adjust=False).mean()
        d["price_ema60"] = d[close_col] / ema - 1.0

        # Momentum pct_change
        d["momentum_10"] = d[close_col].pct_change(momentum_lag)

        # ROC
        d["roc_30"] = ROCIndicator(close=d[close_col], window=roc_30).roc()
        d["roc_60"] = ROCIndicator(close=d[close_col], window=roc_60).roc()

        return d

    df_out = (
        df.groupby(date_col, group_keys=False)
        .apply(per_day, include_groups=False)
    )

    # Robustez: reponer date explícitamente
    df_out[date_col] = pd.to_datetime(df_out.index).date

    return df_out, indicator_cols


def select_final_columns(
    df: pd.DataFrame,
    *,
    date_col: str,
    use_volume: bool,
    indicator_cols: List[str],
    targets: Tuple[int, ...],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_features = ["open", "high", "low", "close"]
    if use_volume:
        base_features.append("volume")

    feature_cols = base_features + indicator_cols
    target_cols = [f"delta_pts_{h}" for h in targets]

    selected = [date_col] + feature_cols + target_cols
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el dataset final: {missing}")

    return df[selected].copy(), feature_cols, target_cols


def apply_dropna_policy(
    df: pd.DataFrame,
    *,
    policy: str,
    target_cols: List[str],
    final_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    policy:
      - "any": dropna sobre todas las columnas finales
      - "targets": dropna solo sobre targets (permite NaNs en features, no recomendado)
    """
    before = int(df.shape[0])

    if policy == "targets":
        out = df.dropna(subset=target_cols)
        rule = "dropna(targets)"
    else:
        out = df.dropna(subset=final_cols)
        rule = "dropna(all_final_cols)"

    after = int(out.shape[0])
    return out, {"dropna_rule": rule, "n_rows_before": before, "n_rows_after": after, "n_rows_dropped": before - after}


def nan_ratios(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cols:
        out[c] = float(df[c].isna().mean())
    return out


# ============================================================
# 6) Summary/Report (envelope estándar)
# ============================================================
def build_stage_04_summary_envelope(
    *,
    paths: Dict[str, Any],
    params: Dict[str, Any],
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    drop_info: Dict[str, Any],
    date_col: str,
) -> Dict[str, Any]:
    n_rows_in = int(df_in.shape[0])
    n_rows_out = int(df_out.shape[0])

    n_days_in = int(df_in[date_col].nunique()) if date_col in df_in.columns else None
    n_days_out = int(df_out[date_col].nunique()) if date_col in df_out.columns else None

    rows_per_day_in = float(df_in.groupby(date_col).size().mean()) if (date_col in df_in.columns and n_rows_in > 0) else None
    rows_per_day_out = float(df_out.groupby(date_col).size().mean()) if (date_col in df_out.columns and n_rows_out > 0) else None

    final_cols = [date_col] + feature_cols + target_cols
    nan_map = nan_ratios(df_out, final_cols)

    envelope = {
        "stage": "stage_04_feature_engineering",
        "created_at_utc": now_utc_iso(),
        "version": "1.0",
        "paths": paths,
        "params": params,
        "metrics": {
            "n_rows_in": float(n_rows_in),
            "n_rows_out": float(n_rows_out),
            "n_rows_dropped": float(drop_info.get("n_rows_dropped", np.nan)),
            "n_days_in": float(n_days_in) if n_days_in is not None else np.nan,
            "n_days_out": float(n_days_out) if n_days_out is not None else np.nan,
            "rows_per_day_mean_in": float(rows_per_day_in) if rows_per_day_in is not None else np.nan,
            "rows_per_day_mean_out": float(rows_per_day_out) if rows_per_day_out is not None else np.nan,
        },
        "details": {
            "schema": {
                "date_col": date_col,
                "features": feature_cols,
                "targets": target_cols,
                "n_features": int(len(feature_cols)),
                "n_targets": int(len(target_cols)),
            },
            "dropna": drop_info,
            "nan_ratio_by_col": {k: round(v, 8) for k, v in nan_map.items()},
        },
    }
    return envelope


def save_json(payload: Dict[str, Any], path: Path) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _wrap_list(items: List[Any], max_line_len: int = 110, indent: str = "  - ") -> str:
    """Imprime listas en líneas no muy largas (amigable para terminal)."""
    s_items = [str(x) for x in (items or [])]
    lines = []
    cur = indent
    for it in s_items:
        part = ("" if cur == indent else ", ") + it
        if len(cur) + len(part) > max_line_len:
            lines.append(cur)
            cur = indent + it
        else:
            cur += part
    if cur != indent:
        lines.append(cur)
    return "\n".join(lines) if lines else (indent + "-")

def print_stage_summary_console(envelope: Dict[str, Any]) -> None:
    m = envelope.get("metrics", {}) or {}

    # schema puede estar en params.schema (recomendado) o en details.schema (fallback)
    schema = (envelope.get("params", {}) or {}).get("schema", {}) or \
             (envelope.get("details", {}) or {}).get("schema", {}) or {}

    features = schema.get("features", []) or []
    targets = schema.get("targets", []) or []

    print("\n" + "=" * 70)
    print(f"STAGE: {envelope.get('stage')}")
    print(f"CREATED_AT_UTC: {envelope.get('created_at_utc')}")
    print(f"VERSION: {envelope.get('version')}")
    print("-" * 70)

    print(f"[FEATURES] n={schema.get('n_features', len(features))} | targets n={schema.get('n_targets', len(targets))}")
    print("[FEATURES_LIST]")
    print(_wrap_list(features))

    print("[TARGETS_LIST]")
    print(_wrap_list(targets))

    print("-" * 70)
    print(
        f"[ROWS] in={int(m.get('n_rows_in', 0))} | "
        f"out={int(m.get('n_rows_out', 0))} | "
        f"dropped={int(m.get('n_rows_dropped', m.get('rows_dropped', 0)))}"
    )
    print(f"[DAYS] in={m.get('n_days_in')} | out={m.get('n_days_out')}")
    print("=" * 70)

# ============================================================
# 7) Tracking MLflow (opcional)
# ============================================================
def mlflow_log_from_envelope(
    envelope: Dict[str, Any],
    *,
    enable: bool,
    run_name: str,
    artifacts: List[Path] | None = None,
) -> None:
    if not enable:
        log.info("[MLFLOW] tracking deshabilitado (enable=False)")
        return

    try:
        import mlflow  # type: ignore
    except Exception as exc:
        log.warning("MLflow no disponible (%s). Omitiendo tracking.", exc)
        return

    params = envelope.get("params", {})
    metrics = envelope.get("metrics", {})

    with mlflow.start_run(run_name=run_name):
        # params (planos)
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        # metrics (numéricas)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        # artifacts
        if artifacts:
            for p in artifacts:
                try:
                    if p.exists():
                        mlflow.log_artifact(str(p))
                except Exception:
                    pass


# ============================================================
# 8) parse_args (override)
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="stage_04_feature_engineering")
    p.add_argument("--in-labeled-parquet", default=str(IN_LABELED_PARQUET))
    p.add_argument("--out-features-parquet", default=str(OUT_FEATURES_PARQUET))
    p.add_argument("--report-summary", default=str(REPORT_SUMMARY))

    p.add_argument("--date-col", default=PARAM_DATE_COL)
    p.add_argument("--close-col", default=PARAM_CLOSE_COL)

    p.add_argument("--use-volume", action="store_true", default=PARAM_USE_VOLUME)

    p.add_argument("--ema-span", type=int, default=PARAM_EMA_SPAN)
    p.add_argument("--momentum-lag", type=int, default=PARAM_MOMENTUM_LAG)
    p.add_argument("--roc-30", type=int, default=PARAM_ROC_30)
    p.add_argument("--roc-60", type=int, default=PARAM_ROC_60)

    p.add_argument("--targets", type=str, default=",".join(str(x) for x in PARAM_TARGETS))
    p.add_argument("--dropna-policy", type=str, default=PARAM_DROPNA_POLICY, choices=["any", "targets"])

    p.add_argument("--enable-mlflow", action="store_true", default=ENABLE_MLFLOW)

    return p.parse_args()


# ============================================================
# 9) main (orquestación)
# ============================================================
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_path = Path(args.in_labeled_parquet)
    out_parquet = Path(args.out_features_parquet)
    report_path = Path(args.report_summary)

    targets = tuple(int(x.strip()) for x in args.targets.split(",") if x.strip())

    # paths envelope
    paths = {
        "inputs": {"labeled_parquet": str(in_path)},
        "outputs": {"features_parquet": str(out_parquet)},
        "reports": {"summary": str(report_path)},
    }

    # params envelope (MLflow-friendly)
    params = {
        "date_col": args.date_col,
        "close_col": args.close_col,
        "use_volume": bool(args.use_volume),
        "ema_span": int(args.ema_span),
        "momentum_lag": int(args.momentum_lag),
        "roc_30": int(args.roc_30),
        "roc_60": int(args.roc_60),
        "targets": list(targets),
        "dropna_policy": str(args.dropna_policy),
    }

    log.info("[1] Cargando labeled parquet: %s", in_path)
    df0 = load_parquet_as_df(in_path)

    log.info("[2] Asegurando columna date=%s", args.date_col)
    df1 = add_or_replace_date_col(df0, date_col=args.date_col)

    log.info("[3] Calculando indicadores por día (sin leakage)")
    df2, indicator_cols = compute_indicators_per_day(
        df1,
        date_col=args.date_col,
        close_col=args.close_col,
        ema_span=args.ema_span,
        momentum_lag=args.momentum_lag,
        roc_30=args.roc_30,
        roc_60=args.roc_60,
    )

    log.info("[4] Seleccionando columnas finales (features + targets)")
    df3, feature_cols, target_cols = select_final_columns(
        df2,
        date_col=args.date_col,
        use_volume=bool(args.use_volume),
        indicator_cols=indicator_cols,
        targets=targets,
    )

    log.info("[5] dropna policy=%s", args.dropna_policy)
    final_cols = [args.date_col] + feature_cols + target_cols
    df_out, drop_info = apply_dropna_policy(
        df3,
        policy=args.dropna_policy,
        target_cols=target_cols,
        final_cols=final_cols,
    )
    log.info("[5.1] %s | %s -> %s (dropped=%s)",
             drop_info["dropna_rule"], drop_info["n_rows_before"], drop_info["n_rows_after"], drop_info["n_rows_dropped"])

    log.info("[6] Guardando features parquet: %s", out_parquet)
    _ensure_parent_dir(out_parquet)
    df_out.to_parquet(out_parquet, index=True)

    log.info("[7] Construyendo summary envelope y guardando: %s", report_path)
    envelope = build_stage_04_summary_envelope(
        paths=paths,
        params=params,
        df_in=df1,
        df_out=df_out,
        feature_cols=feature_cols,
        target_cols=target_cols,
        drop_info=drop_info,
        date_col=args.date_col,
    )
    save_json(envelope, report_path)

    print_stage_summary_console(envelope)

    log.info("[8] MLflow tracking (enable=%s)", bool(args.enable_mlflow))
    mlflow_log_from_envelope(
        envelope,
        enable=bool(args.enable_mlflow),
        run_name="stage_04_feature_engineering",
        artifacts=[report_path],
    )

    log.info("[OK] stage_04_feature_engineering finalizado correctamente.")


# ============================================================
# 10) Boilerplate
# ============================================================
if __name__ == "__main__":
    main()
