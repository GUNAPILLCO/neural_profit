# ============================================================
# stage_05_time_aware_data_splitting.py
#
# Propósito:
#   Split temporal por jornadas (train/valid/test) sin mezclar días.
#
# Inputs:
#   - IN_FEATURES_PARQUET: data/features/mnq_features_target.parquet
#   - IN_SCHEMA_SUMMARY  : reports/features_target_summary.json  (schema.features/targets)
#   - PARAMS_YAML        : params.yaml (stage_05.*)
#
# Outputs:
#   - OUT_TRAIN_PARQUET  : data/splits/mnq_train.parquet
#   - OUT_VALID_PARQUET  : data/splits/mnq_valid.parquet
#   - OUT_TEST_PARQUET   : data/splits/mnq_test.parquet
#   - OUT_SPLITS_JSON    : data/splits/splits.json  (días por split)
#
# Report (envelope estándar):
#   - REPORT_SUMMARY     : reports/stage_05_time_aware_data_splitting_summary.json
#
# Params (params.yaml: stage_05):
#   - train_ratio: float
#   - valid_ratio: float
#   - expected_gap_minutes: int
#
# Notas:
#   - El split es por DÍAS (columna date). No hay shuffle.
#   - Si hay NaNs o gaps temporales dentro de una jornada -> FAIL.
# ============================================================

from __future__ import annotations

# =========================
# 1) Imports (ordenados)
# =========================
import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# =========================
# 2) Logging (uniforme)
# =========================
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_05_time_aware_data_splitting")

# =========================
# 3) Configuración DVC-friendly (SIEMPRE presente)
# =========================
IN_FEATURES_PARQUET = Path(os.environ.get("IN_FEATURES_PARQUET", "data/features/mnq_features_target.parquet"))
IN_SCHEMA_SUMMARY = Path(os.environ.get("IN_SCHEMA_SUMMARY", "reports/stage_04_feature_engineering_summary.json"))
PARAMS_YAML = Path(os.environ.get("PARAMS_YAML", "params.yaml"))

OUT_TRAIN_PARQUET = Path(os.environ.get("OUT_TRAIN_PARQUET", "data/splits/mnq_train.parquet"))
OUT_VALID_PARQUET = Path(os.environ.get("OUT_VALID_PARQUET", "data/splits/mnq_valid.parquet"))
OUT_TEST_PARQUET = Path(os.environ.get("OUT_TEST_PARQUET", "data/splits/mnq_test.parquet"))
OUT_SPLITS_JSON = Path(os.environ.get("OUT_SPLITS_JSON", "data/splits/splits.json"))

REPORT_SUMMARY = Path(
    os.environ.get("REPORT_SUMMARY", "reports/stage_05_time_aware_data_splitting_summary.json")
)

# MLflow flag (opcional)
ENABLE_MLFLOW = bool(int(os.environ.get("ENABLE_MLFLOW", "0")))


# =========================
# 4) Params (defaults reproducibles)
# =========================
@dataclass(frozen=True)
class Stage05Params:
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    expected_gap_minutes: int = 1


# =========================
# 5) Utilidades puras
# =========================
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró YAML: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró parquet: {path}")
    return pd.read_parquet(path)


def add_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out[date_col] = out.index.date
    cols = [date_col] + [c for c in out.columns if c != date_col]
    return out[cols]


def load_features_targets_from_summary(path: Path) -> Tuple[List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró schema summary: {path}")

    with path.open("r", encoding="utf-8") as f:
        s: Dict[str, Any] = json.load(f)

    # 1) Intenta formato "envelope": details.schema
    schema = None
    details = s.get("details")
    if isinstance(details, dict):
        ds = details.get("schema")
        if isinstance(ds, dict):
            schema = ds

    # 2) Fallback a formato simple: schema
    if schema is None:
        sch = s.get("schema")
        if isinstance(sch, dict):
            schema = sch

    if not isinstance(schema, dict):
        raise ValueError(
            f"Summary inválido: no se encontró 'details.schema' ni 'schema'. "
            f"Keys disponibles: {sorted(s.keys())}"
        )

    features = schema.get("features", [])
    targets = schema.get("targets", [])

    if not isinstance(features, list) or not isinstance(targets, list):
        raise ValueError(
            f"schema.features/targets inválidos. "
            f"types: features={type(features).__name__}, targets={type(targets).__name__}"
        )

    # Normaliza a strings (por seguridad)
    features = [str(x) for x in features]
    targets = [str(x) for x in targets]

    return features, targets


def read_stage05_params(params_yaml: Path) -> Stage05Params:
    p = load_yaml(params_yaml)
    s5 = p.get("stage_05", {}) or {}
    return Stage05Params(
        train_ratio=float(s5.get("train_ratio", Stage05Params.train_ratio)),
        valid_ratio=float(s5.get("valid_ratio", Stage05Params.valid_ratio)),
        expected_gap_minutes=int(s5.get("expected_gap_minutes", Stage05Params.expected_gap_minutes)),
    )


def validate_split_ratios(train_ratio: float, valid_ratio: float) -> float:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio inválido. Requiere 0 < train_ratio < 1.")
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio inválido. Requiere 0 < valid_ratio < 1.")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("Ratios inválidos: train_ratio + valid_ratio debe ser < 1.")
    return float(1.0 - train_ratio - valid_ratio)


def validate_no_nans(df: pd.DataFrame, cols: Sequence[str], date_col: str = "date") -> Dict[str, Any]:
    """
    Validación dura: no se permiten NaNs en columnas finales.
    Devuelve stats para reporte si pasa.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para validar NaNs: {missing}")

    # Conteo total
    total_nans = int(df[cols].isna().sum().sum())
    if total_nans > 0:
        # detalle compacto: por columna y por día (top)
        by_col = df[cols].isna().sum().sort_values(ascending=False)
        bad_cols = by_col[by_col > 0]

        # por día: cantidad total de NaNs en columnas finales
        per_day = df.groupby(date_col, sort=False)[cols].apply(lambda x: int(x.isna().sum().sum()))
        per_day = per_day[per_day > 0].sort_values(ascending=False)

        raise RuntimeError(
            "[ERROR] Se detectaron NaNs en el dataset.\n"
            f"- total_nans={total_nans}\n"
            f"- cols_con_nans={bad_cols.to_dict()}\n"
            f"- days_con_nans_top={per_day.head(10).to_dict()}"
        )

    return {
        "total_nans": 0,
        "nan_ratio_total": 0.0,
    }


def validate_no_gaps(df: pd.DataFrame, gap_minutes: int, date_col: str = "date") -> Dict[str, Any]:
    """
    Validación dura: dentro de cada día, el diff entre timestamps consecutivos debe ser gap_minutes.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser DatetimeIndex para validar gaps.")

    expected = pd.Timedelta(minutes=int(gap_minutes))
    gap_events: List[Dict[str, Any]] = []

    # OJO: diff por día (para no contaminar el diff entre días)
    for day, g in df.groupby(date_col, sort=False):
        idx = g.index.sort_values()
        if len(idx) <= 1:
            continue
        diffs = idx.to_series().diff().iloc[1:]
        bad = diffs[diffs != expected]
        if not bad.empty:
            for ts, d in bad.items():
                gap_events.append({"date": str(day), "timestamp": str(ts), "time_diff": str(d)})

    if gap_events:
        # limitar para no explotar logs
        preview = gap_events[:50]
        raise RuntimeError(
            "[ERROR] Se detectaron gaps temporales.\n"
            f"- expected={expected}\n"
            f"- n_gaps={len(gap_events)}\n"
            f"- preview={preview}"
        )

    return {"n_gaps": 0, "expected_gap_minutes": int(gap_minutes)}


def split_days(unique_days: pd.Index, train_ratio: float, valid_ratio: float) -> Tuple[pd.Index, pd.Index, pd.Index]:
    n_days = int(len(unique_days))
    if n_days <= 0:
        raise RuntimeError("No se encontraron días (columna date vacía).")

    n_train = int(n_days * train_ratio)
    n_valid = int(n_days * valid_ratio)
    n_test = n_days - n_train - n_valid

    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(f"Split inválido por cantidad de días: total={n_days}, train={n_train}, valid={n_valid}, test={n_test}")

    train_days = unique_days[:n_train]
    valid_days = unique_days[n_train : n_train + n_valid]
    test_days = unique_days[n_train + n_valid :]

    return train_days, valid_days, test_days


def build_splits_payload(train_days: pd.Index, valid_days: pd.Index, test_days: pd.Index) -> Dict[str, Any]:
    return {
        "train_days": [str(d) for d in train_days],
        "valid_days": [str(d) for d in valid_days],
        "test_days": [str(d) for d in test_days],
        "n_train_days": int(len(train_days)),
        "n_valid_days": int(len(valid_days)),
        "n_test_days": int(len(test_days)),
    }


def _stats_block(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    n_rows = int(df.shape[0])
    n_days = int(df[date_col].nunique()) if (date_col in df.columns and n_rows > 0) else 0
    rows_per_day = float(df.groupby(date_col).size().mean()) if (date_col in df.columns and n_rows > 0) else np.nan
    return {
        "n_rows": n_rows,
        "n_days": n_days,
        "rows_per_day_mean": None if not np.isfinite(rows_per_day) else round(rows_per_day, 2),
        "datetime_min": None if n_rows == 0 else str(df.index.min()),
        "datetime_max": None if n_rows == 0 else str(df.index.max()),
    }


def build_stage05_envelope(
    *,
    created_at_utc: str,
    version: str,
    paths: Dict[str, Any],
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": "stage_05_time_aware_data_splitting",
        "created_at_utc": created_at_utc,
        "version": version,
        "paths": paths,
        "params": params,
        "metrics": metrics,
        "details": details,
    }


def print_stage_summary_console(envelope: Dict[str, Any]) -> None:
    m = envelope.get("metrics", {})
    schema = envelope.get("details", {}).get("schema", {})
    splits = envelope.get("details", {}).get("splits", {})
    ratios = envelope.get("params", {}).get("ratios", {})

    print("\n" + "=" * 70)
    print(f"STAGE: {envelope.get('stage')}")
    print(f"CREATED_AT_UTC: {envelope.get('created_at_utc')}")
    print(f"VERSION: {envelope.get('version')}")
    print("-" * 70)

    # Mostrar features/targets explícitamente
    features = schema.get("features", [])
    targets = schema.get("targets", [])
    print(f"[FEATURES] n={schema.get('n_features')} | {features}")
    print(f"[TARGETS ] n={schema.get('n_targets')} | {targets}")

    print("-" * 70)
    print(f"[RATIOS] train={ratios.get('train_ratio')} | valid={ratios.get('valid_ratio')} | test={ratios.get('test_ratio')}")
    print(f"[GAPS] expected_gap_minutes={m.get('expected_gap_minutes')} | n_gaps={m.get('n_gaps')}")
    print(f"[NANS] total_nans={m.get('total_nans')}")

    def _one(name: str) -> str:
        b = splits.get(name, {})
        return f"{name}: rows={b.get('n_rows')} | days={b.get('n_days')} | dt=[{b.get('datetime_min')} .. {b.get('datetime_max')}]"

    print("-" * 70)
    print(_one("input"))
    print(_one("train"))
    print(_one("valid"))
    print(_one("test"))
    print("=" * 70)


# =========================
# 6) MLflow (opcional)
# =========================
def mlflow_log_from_envelope(envelope: Dict[str, Any], *, enable: bool, run_name: str, artifacts: Optional[List[str]] = None) -> None:
    if not enable:
        log.info("[MLflow] enable=False. Saltando.")
        return

    try:
        import mlflow  # type: ignore
    except Exception:
        log.info("[MLflow] mlflow no disponible. Saltando.")
        return

    params = envelope.get("params", {})
    metrics = envelope.get("metrics", {})

    with mlflow.start_run(run_name=run_name):
        # params: solo tipos simples
        def _log_param(k: str, v: Any) -> None:
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)

        # aplanar ratios
        ratios = (params.get("ratios", {}) or {})
        _log_param("train_ratio", ratios.get("train_ratio"))
        _log_param("valid_ratio", ratios.get("valid_ratio"))
        _log_param("test_ratio", ratios.get("test_ratio"))
        _log_param("expected_gap_minutes", params.get("expected_gap_minutes"))

        # métricas numéricas comparables
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))

        # artifacts
        if artifacts:
            for p in artifacts:
                try:
                    if p and os.path.exists(p):
                        mlflow.log_artifact(p)
                except Exception:
                    pass


# =========================
# 7) parse_args (override)
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-parquet", default=str(IN_FEATURES_PARQUET))
    p.add_argument("--in-schema-summary", default=str(IN_SCHEMA_SUMMARY))
    p.add_argument("--params-yaml", default=str(PARAMS_YAML))

    p.add_argument("--out-train-parquet", default=str(OUT_TRAIN_PARQUET))
    p.add_argument("--out-valid-parquet", default=str(OUT_VALID_PARQUET))
    p.add_argument("--out-test-parquet", default=str(OUT_TEST_PARQUET))
    p.add_argument("--out-splits-json", default=str(OUT_SPLITS_JSON))
    p.add_argument("--report-summary", default=str(REPORT_SUMMARY))

    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--valid-ratio", type=float, default=None)
    p.add_argument("--expected-gap-minutes", type=int, default=None)

    p.add_argument("--enable-mlflow", action="store_true")
    return p.parse_args()


# =========================
# 8) main (orquestación)
# =========================
def main() -> None:
    log.info("[0] Parseando argumentos (CLI/env)")
    args = parse_args()

    in_parquet = Path(args.in_parquet)
    in_schema = Path(args.in_schema_summary)
    params_yaml = Path(args.params_yaml)

    out_train = Path(args.out_train_parquet)
    out_valid = Path(args.out_valid_parquet)
    out_test = Path(args.out_test_parquet)
    out_splits = Path(args.out_splits_json)
    report_summary = Path(args.report_summary)

    # Params desde params.yaml (y override si viene por CLI)
    log.info("[1] Leyendo params stage_05 desde: %s", params_yaml)
    p = read_stage05_params(params_yaml)

    train_ratio = float(args.train_ratio) if args.train_ratio is not None else p.train_ratio
    valid_ratio = float(args.valid_ratio) if args.valid_ratio is not None else p.valid_ratio
    expected_gap_minutes = int(args.expected_gap_minutes) if args.expected_gap_minutes is not None else p.expected_gap_minutes
    test_ratio = validate_split_ratios(train_ratio, valid_ratio)

    log.info("[2] Cargando dataset features/targets: %s", in_parquet)
    df_in = load_parquet(in_parquet)
    df_in = add_date_column(df_in, date_col="date").sort_index()

    log.info("[3] Cargando schema (features/targets) desde: %s", in_schema)
    features, targets = load_features_targets_from_summary(in_schema)

    final_cols = ["date"] + features + targets

    log.info("[4] Validaciones duras (NaNs + gaps)")
    nan_stats = validate_no_nans(df_in, cols=final_cols, date_col="date")
    gap_stats = validate_no_gaps(df_in, gap_minutes=expected_gap_minutes, date_col="date")

    log.info("[5] Split temporal por días (sin shuffle)")
    unique_days = pd.Index(sorted(df_in["date"].unique()))
    train_days, valid_days, test_days = split_days(unique_days, train_ratio=train_ratio, valid_ratio=valid_ratio)

    df_train = df_in[df_in["date"].isin(train_days)].copy().sort_index()
    df_valid = df_in[df_in["date"].isin(valid_days)].copy().sort_index()
    df_test = df_in[df_in["date"].isin(test_days)].copy().sort_index()

    log.info("[6] Guardando parquets train/valid/test")
    _ensure_parent_dir(out_train)
    _ensure_parent_dir(out_valid)
    _ensure_parent_dir(out_test)
    df_train.to_parquet(out_train, index=True)
    df_valid.to_parquet(out_valid, index=True)
    df_test.to_parquet(out_test, index=True)

    log.info("[7] Guardando splits.json (artifact operativo)")
    splits_payload = build_splits_payload(train_days, valid_days, test_days)
    save_json(out_splits, splits_payload)

    # Envelope summary
    log.info("[8] Construyendo report summary (envelope estándar)")
    created_at_utc = _utc_now_iso()

    paths = {
        "inputs": {
            "features_parquet": str(in_parquet),
            "schema_summary": str(in_schema),
            "params_yaml": str(params_yaml),
        },
        "outputs": {
            "train_parquet": str(out_train),
            "valid_parquet": str(out_valid),
            "test_parquet": str(out_test),
            "splits_json": str(out_splits),
        },
        "reports": {
            "summary": str(report_summary),
        },
    }

    params_block = {
        "ratios": {
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": round(test_ratio, 6),
        },
        "expected_gap_minutes": expected_gap_minutes,
    }

    details = {
        "schema": {
            "date_col": "date",
            "features": features,
            "targets": targets,
            "n_features": int(len(features)),
            "n_targets": int(len(targets)),
        },
        "splits": {
            "input": _stats_block(df_in, date_col="date"),
            "train": _stats_block(df_train, date_col="date"),
            "valid": _stats_block(df_valid, date_col="date"),
            "test": _stats_block(df_test, date_col="date"),
        },
        "split_days": splits_payload,
    }

    metrics = {
        "n_rows_in": float(df_in.shape[0]),
        "n_rows_train": float(df_train.shape[0]),
        "n_rows_valid": float(df_valid.shape[0]),
        "n_rows_test": float(df_test.shape[0]),
        "n_days_in": float(df_in["date"].nunique()),
        "n_days_train": float(df_train["date"].nunique()),
        "n_days_valid": float(df_valid["date"].nunique()),
        "n_days_test": float(df_test["date"].nunique()),
        "total_nans": float(nan_stats.get("total_nans", np.nan)),
        "n_gaps": float(gap_stats.get("n_gaps", np.nan)),
        "expected_gap_minutes": float(expected_gap_minutes),
    }

    envelope = build_stage05_envelope(
        created_at_utc=created_at_utc,
        version="1.0",
        paths=paths,
        params=params_block,
        metrics=metrics,
        details=details,
    )

    save_json(report_summary, envelope)
    print_stage_summary_console(envelope)

    # MLflow (opcional)
    log.info("[9] MLflow tracking (enable=%s)", bool(args.enable_mlflow or ENABLE_MLFLOW))
    mlflow_log_from_envelope(
        envelope,
        enable=bool(args.enable_mlflow or ENABLE_MLFLOW),
        run_name="stage_05_time_aware_data_splitting",
        artifacts=[str(report_summary), str(out_splits)],
    )

    log.info("[DONE] stage_05_time_aware_data_splitting finalizado correctamente.")


if __name__ == "__main__":
    main()
