# ============================================================
# stage_06_window_scaling_seq2seq.py
# STAGE_06 – WINDOW SCALING & FINAL DATASET ASSEMBLY (SEQ2SEQ MNQ)
#
# Contrato:
#   - Lee splits: data/splits/mnq_{train,valid,test}.parquet
#   - Lee schema (features/targets): reports/stage_04_feature_engineering_summary.json
#   - Lee gestation window: reports/stage_03b_target_definition_summary.json
#   - Filtra cada split por la ventana de gestación (sin mezclar días)
#   - Construye 1 ventana SEQ2SEQ por día (X: [W,F], y: [W])
#   - Guarda windows raw (.npz) y windows escaladas (.npz) + scaler (.pkl) por horizonte
#   - Genera reporte JSON (envelope) con shapes, checks y stats
#
# Salidas:
#   - data/windows/windows_{split}_{h}.npz
#   - data/windows/scaled/windows_{split}_{h}_z.npz
#   - data/windows/scaled/scaler_{h}.pkl
#   - reports/stage_06_window_scaling_seq2seq_summary.json
# ============================================================

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_06")

# ----------------------------
# MLflow (opcional)
# ----------------------------
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None


# ---------------------------------------------------------------------
# Configuración de rutas (DVC-friendly)
# ---------------------------------------------------------------------
# Estas rutas son RELATIVAS al repositorio.
# DVC necesita paths locales y determinísticos.
# Si mañana quiere apuntar a Drive, se hace vía DVC remote o symlink,
# NO cambiando la lógica del stage.
IN_SPLIT_TRAIN = Path(os.environ.get("IN_SPLIT_TRAIN", "data/splits/mnq_train.parquet"))
IN_SPLIT_VALID = Path(os.environ.get("IN_SPLIT_VALID", "data/splits/mnq_valid.parquet"))
IN_SPLIT_TEST = Path(os.environ.get("IN_SPLIT_TEST", "data/splits/mnq_test.parquet"))

IN_STAGE04_SUMMARY = Path(
    os.environ.get("IN_STAGE04_SUMMARY", "reports/stage_04_feature_engineering_summary.json")
)
IN_STAGE03B_SUMMARY = Path(
    os.environ.get("IN_STAGE03B_SUMMARY", "reports/stage_03b_target_definition_summary.json")
)

PARAMS_YAML = Path(os.environ.get("PARAMS_YAML", "params.yaml"))

OUT_WINDOWS_DIR = Path(os.environ.get("OUT_WINDOWS_DIR", "data/windows"))
OUT_SCALED_DIR = Path(os.environ.get("OUT_SCALED_DIR", "data/windows/scaled"))

REPORT_SUMMARY = Path(
    os.environ.get("REPORT_SUMMARY", "reports/stage_06_window_scaling_seq2seq_summary.json")
)

# Defaults si params.yaml no está / faltan claves
DEFAULT_HORIZONS = [60, 90]
DEFAULT_SCALER_TYPE = "standard"
DEFAULT_TIMEZONE_STR = "America/New_York"
DEFAULT_DATE_COL = "date"

# ---------------------------------------------------------------------
# Helpers IO
# ---------------------------------------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Se esperaba dict en {path}, llegó: {type(obj)}")
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_params_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        return {}
    return obj


# ---------------------------------------------------------------------
# Params stage_06
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Stage06Params:
    date_col: str
    horizons: List[int]
    scaler_type: str
    timezone_str: str
    # criterio de features por horizonte (si quiere mantener exclusiones)
    drop_feature_h60: str | None
    drop_feature_h90: str | None


def read_stage06_params(params: Dict[str, Any]) -> Stage06Params:
    s6 = (params.get("stage_06") or {})
    date_col = str(s6.get("date_col", DEFAULT_DATE_COL))
    horizons = s6.get("horizons", DEFAULT_HORIZONS)
    horizons = [int(h) for h in horizons]

    scaler_type = str(s6.get("scaler_type", DEFAULT_SCALER_TYPE))
    timezone_str = str(s6.get("timezone_str", DEFAULT_TIMEZONE_STR))

    # Mantener su criterio original: h60 sin roc_30, h90 sin roc_60 (configurable)
    drop_feature_h60 = s6.get("drop_feature_h60", "roc_30")
    drop_feature_h90 = s6.get("drop_feature_h90", "roc_60")

    return Stage06Params(
        date_col=date_col,
        horizons=horizons,
        scaler_type=scaler_type,
        timezone_str=timezone_str,
        drop_feature_h60=str(drop_feature_h60) if drop_feature_h60 else None,
        drop_feature_h90=str(drop_feature_h90) if drop_feature_h90 else None,
    )


# ---------------------------------------------------------------------
# Carga split parquet
# ---------------------------------------------------------------------
def load_mnq_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")
    log.info("[OK] Cargando parquet: %s", path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ---------------------------------------------------------------------
# Schema desde stage_04 (features/targets)
# ---------------------------------------------------------------------
def load_features_targets_from_stage04(summary_path: Path) -> Tuple[List[str], List[str]]:
    env = load_json_dict(summary_path)
    schema = (env.get("details", {}) or {}).get("schema", {}) or {}

    features = schema.get("features")
    targets = schema.get("targets")

    if not isinstance(features, list) or not features:
        raise KeyError("No se encontró 'details.schema.features' (lista no vacía) en stage_04 summary.")
    if not isinstance(targets, list) or len(targets) < 2:
        raise KeyError("No se encontró 'details.schema.targets' con al menos 2 elementos en stage_04 summary.")

    return [str(x) for x in features], [str(x) for x in targets]


def build_horizon_feature_target_map(
    features: List[str],
    targets: List[str],
    p: Stage06Params,
) -> Dict[int, Dict[str, Any]]:
    # targets esperados: delta_pts_60 y delta_pts_90 (por su stage_04)
    by_h: Dict[int, Dict[str, Any]] = {}

    # helper: busca target que termine con _{h}
    def _find_target(h: int) -> str:
        suffix = f"_{h}"
        for t in targets:
            if t.endswith(suffix):
                return t
        # fallback: por posición si hay [60,90] en ese orden
        if len(targets) >= 2 and h == 60:
            return targets[0]
        if len(targets) >= 2 and h == 90:
            return targets[1]
        raise KeyError(f"No se pudo resolver target para horizonte={h} desde targets={targets}")

    for h in p.horizons:
        feats_h = list(features)
        if h == 60 and p.drop_feature_h60 and p.drop_feature_h60 in feats_h:
            feats_h = [f for f in feats_h if f != p.drop_feature_h60]
        if h == 90 and p.drop_feature_h90 and p.drop_feature_h90 in feats_h:
            feats_h = [f for f in feats_h if f != p.drop_feature_h90]

        by_h[h] = {
            "features": feats_h,
            "target": _find_target(h),
        }

    return by_h


# ---------------------------------------------------------------------
# Gestation window desde stage_03b
# ---------------------------------------------------------------------
def get_gestation_window_from_stage03b_summary(stage03b_env: Dict[str, Any]) -> Tuple[str, str]:
    details = stage03b_env.get("details", {}) or {}
    gw = details.get("gestation_window", {}) or {}

    start = gw.get("start_hhmm")
    end = gw.get("end_hhmm")
    if start and end:
        return str(start), str(end)

    start = gw.get("start") or details.get("gestation_window_start")
    end = gw.get("end") or details.get("gestation_window_end")
    if start and end:
        return str(start), str(end)

    mod_min = gw.get("minute_of_day_min")
    mod_max = gw.get("minute_of_day_max")
    if mod_min is not None and mod_max is not None:
        mod_min = int(mod_min)
        mod_max = int(mod_max)
        start = f"{mod_min // 60:02d}:{mod_min % 60:02d}"
        end = f"{mod_max // 60:02d}:{mod_max % 60:02d}"
        return start, end

    raise KeyError(
        "No encontré gestation_window en stage_03b summary. "
        "Se esperaba details.gestation_window.start_hhmm/end_hhmm o minute_of_day_min/max."
    )


# ---------------------------------------------------------------------
# Filtrado por ventana de gestación (sin mezclar días)
# ---------------------------------------------------------------------
def filter_gestation_window(df: pd.DataFrame, start: str, end: str, date_col: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index debe ser DatetimeIndex.")
    if date_col not in df.columns:
        raise KeyError(f"No se encontró '{date_col}' en el DataFrame.")

    df = df.sort_index()

    # check: date_col coincide con index.date
    idx_dates = df.index.date
    col_dates = pd.to_datetime(df[date_col]).dt.date
    if not (col_dates.values == idx_dates).all():
        raise ValueError(f"Inconsistencia entre '{date_col}' y la fecha del DatetimeIndex.")

    out = df.between_time(start, end, inclusive="both").sort_index()

    out_idx_dates = out.index.date
    out_col_dates = pd.to_datetime(out[date_col]).dt.date
    if not (out_col_dates.values == out_idx_dates).all():
        raise ValueError("El filtrado generó inconsistencia entre date_col e índice.")

    return out


def find_window_size(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, date_col: str) -> int:
    n_train = int(df_train.groupby(date_col).size().median())
    n_valid = int(df_valid.groupby(date_col).size().median())
    n_test = int(df_test.groupby(date_col).size().median())

    if n_train == n_valid == n_test:
        return n_train
    raise ValueError(f"Window size mismatch (median rows/day): train={n_train}, valid={n_valid}, test={n_test}")


# ---------------------------------------------------------------------
# Construcción ventanas SEQ2SEQ (1 por día)
# ---------------------------------------------------------------------
def build_daily_seq2seq_windows(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    window_size: int,
    date_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for _, g in df.groupby(date_col):
        g = g.sort_index()
        if len(g) != window_size:
            continue

        X_day = g[features].to_numpy()
        y_day = g[target_col].to_numpy()

        if np.isnan(X_day).any() or np.isnan(y_day).any():
            continue

        X_list.append(X_day)
        y_list.append(y_day)

    if not X_list:
        return np.empty((0, window_size, len(features))), np.empty((0, window_size))

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def save_windows_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    _ensure_parent_dir(path)
    np.savez_compressed(path, X=X, y=y)


def load_windows_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


def prepare_or_load_windows(
    split_name: str,
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    window_size: int,
    date_col: str,
    out_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if out_path.exists():
        log.info("[OK] Cargando windows %s desde: %s", split_name, out_path)
        return load_windows_npz(out_path)

    log.info("[GEN] Generando windows %s (%s) -> %s", split_name, target_col, out_path)
    X, y = build_daily_seq2seq_windows(df, features, target_col, window_size, date_col)
    save_windows_npz(out_path, X, y)
    log.info("[OK] Guardado: %s | X=%s | y=%s", out_path, X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------
# Escalado (solo X)
# ---------------------------------------------------------------------
def choose_scaler(scaler_type: str):
    st = scaler_type.lower()
    if st in {"standard", "z", "zscore"}:
        return StandardScaler()
    if st in {"minmax", "min_max"}:
        return MinMaxScaler()
    raise ValueError("scaler_type debe ser 'standard' o 'minmax'")


def fit_scaler_on_train_3d(X_train: np.ndarray, scaler) -> Any:
    if X_train.ndim != 3:
        raise ValueError(f"X_train debe ser 3D [N,W,F]. Llegó: {X_train.shape}")
    _, _, F = X_train.shape
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def transform_3d(X: np.ndarray, scaler) -> np.ndarray:
    n, W, F = X.shape
    Xf = X.reshape(-1, F)
    return scaler.transform(Xf).reshape(n, W, F)


def scale_and_save_windows(
    *,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    scaler_type: str,
    scaler_path: Path,
    out_train: Path,
    out_valid: Path,
    out_test: Path,
) -> None:
    scaler = choose_scaler(scaler_type)
    scaler = fit_scaler_on_train_3d(X_train, scaler)

    X_train_s = transform_3d(X_train, scaler)
    X_valid_s = transform_3d(X_valid, scaler)
    X_test_s = transform_3d(X_test, scaler)

    _ensure_parent_dir(scaler_path)
    joblib.dump(scaler, scaler_path)

    save_windows_npz(out_train, X_train_s, y_train)
    save_windows_npz(out_valid, X_valid_s, y_valid)
    save_windows_npz(out_test, X_test_s, y_test)


# ---------------------------------------------------------------------
# Reporte (envelope) + pretty print
# ---------------------------------------------------------------------
def flatten_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def npz_info(path: Path) -> Dict[str, Any]:
    X, y = load_windows_npz(path)
    return {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "has_nan_X": bool(np.isnan(X).any()) if X.size else False,
        "has_nan_y": bool(np.isnan(y).any()) if y.size else False,
    }


def build_stage06_envelope(
    *,
    created_at_utc: str,
    version: str,
    paths: Dict[str, Any],
    params_used: Dict[str, Any],
    metrics: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": "stage_06_window_scaling_seq2seq",
        "created_at_utc": created_at_utc,
        "version": version,
        "paths": paths,
        "params": params_used,
        "metrics": metrics,
        "details": details,
    }


def print_stage_summary_console(envelope: Dict[str, Any]) -> None:
    m = envelope.get("metrics", {}) or {}
    d = envelope.get("details", {}) or {}
    cfg = d.get("config", {}) or {}

    print("\n" + "=" * 78)
    print(f"STAGE: {envelope.get('stage')}")
    print(f"CREATED_AT_UTC: {envelope.get('created_at_utc')}")
    print(f"VERSION: {envelope.get('version')}")
    print("-" * 78)
    print(f"[CONFIG] window_size={cfg.get('window_size')} | scaler={cfg.get('scaler_type')} | horizons={cfg.get('horizons')}")
    print(f"[WINDOW] {cfg.get('gestation_window_start')} -> {cfg.get('gestation_window_end')}")
    print(f"[DAYS] train={m.get('n_days_train')} | valid={m.get('n_days_valid')} | test={m.get('n_days_test')}")
    print("=" * 78)


# ---------------------------------------------------------------------
# MLflow summary (params + metrics)
# ---------------------------------------------------------------------
def build_mlflow_payload(envelope: Dict[str, Any]) -> Dict[str, Any]:
    p = envelope.get("params", {}) or {}
    m = envelope.get("metrics", {}) or {}
    # MLflow params deben ser simples (str/int/float/bool). Lists -> str(json)
    params_out: Dict[str, Any] = {}
    for k, v in p.items():
        if isinstance(v, (dict, list)):
            params_out[k] = json.dumps(v, ensure_ascii=False)
        else:
            params_out[k] = v
    # metrics solo numéricos
    metrics_out: Dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            metrics_out[k] = float(v)
    return {"params": params_out, "metrics": metrics_out}


def log_to_mlflow(payload: Dict[str, Any], *, run_name: str, artifacts: List[Path], enable: bool) -> None:
    if (not enable) or (mlflow is None):
        log.info("[MLflow] tracking (enable=%s)", enable)
        return

    with mlflow.start_run(run_name=run_name):
        for k, v in payload.get("params", {}).items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        for k, v in payload.get("metrics", {}).items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass

        for p in artifacts:
            try:
                if p.exists():
                    mlflow.log_artifact(str(p))
            except Exception:
                pass


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="stage_06_window_scaling_seq2seq")
    ap.add_argument("--params-yaml", type=str, default=str(PARAMS_YAML))

    ap.add_argument("--in-train", type=str, default=str(IN_SPLIT_TRAIN))
    ap.add_argument("--in-valid", type=str, default=str(IN_SPLIT_VALID))
    ap.add_argument("--in-test", type=str, default=str(IN_SPLIT_TEST))
    ap.add_argument("--in-stage04-summary", type=str, default=str(IN_STAGE04_SUMMARY))
    ap.add_argument("--in-stage03b-summary", type=str, default=str(IN_STAGE03B_SUMMARY))

    ap.add_argument("--out-windows-dir", type=str, default=str(OUT_WINDOWS_DIR))
    ap.add_argument("--out-scaled-dir", type=str, default=str(OUT_SCALED_DIR))
    ap.add_argument("--report-summary", type=str, default=str(REPORT_SUMMARY))

    ap.add_argument("--scaler-type", type=str, default="")  # override
    ap.add_argument("--enable-mlflow", action="store_true")
    return ap.parse_args()


# ---------------------------------------------------------------------
# Main (orquestador)
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # 0) params.yaml (para stage_06)
    params = load_params_yaml(Path(args.params_yaml))
    p6 = read_stage06_params(params)

    scaler_type = args.scaler_type.strip() or p6.scaler_type
    mlflow_enable = bool(args.enable_mlflow) or (os.environ.get("MLFLOW_ENABLE", "0") == "1")

    # 1) Cargar splits
    log.info("[1] Cargando splits parquet")
    df_train = load_mnq_parquet(Path(args.in_train))
    df_valid = load_mnq_parquet(Path(args.in_valid))
    df_test = load_mnq_parquet(Path(args.in_test))

    # 2) Cargar features/targets desde stage_04 summary
    log.info("[2] Cargando schema desde stage_04 summary")
    features_all, targets_all = load_features_targets_from_stage04(Path(args.in_stage04_summary))
    map_by_h = build_horizon_feature_target_map(features_all, targets_all, p6)

    # 3) Cargar gestation window desde stage_03b summary
    log.info("[3] Cargando gestation window desde stage_03b summary")
    s3b = load_json_dict(Path(args.in_stage03b_summary))
    gw_start, gw_end = get_gestation_window_from_stage03b_summary(s3b)
    log.info("[OK] Gestation window: %s -> %s", gw_start, gw_end)

    # 4) Filtrar por ventana
    log.info("[4] Filtrando ventana de gestación en cada split")
    df_train_g = filter_gestation_window(df_train, gw_start, gw_end, date_col=p6.date_col)
    df_valid_g = filter_gestation_window(df_valid, gw_start, gw_end, date_col=p6.date_col)
    df_test_g = filter_gestation_window(df_test, gw_start, gw_end, date_col=p6.date_col)

    # 5) window_size (mediana rows/day, debe coincidir)
    log.info("[5] Determinando window_size")
    window_size = find_window_size(df_train_g, df_valid_g, df_test_g, date_col=p6.date_col)
    log.info("[OK] window_size=%d", window_size)

    out_windows_dir = Path(args.out_windows_dir)
    out_scaled_dir = Path(args.out_scaled_dir)
    report_path = Path(args.report_summary)

    _ensure_parent_dir(report_path)
    out_windows_dir.mkdir(parents=True, exist_ok=True)
    out_scaled_dir.mkdir(parents=True, exist_ok=True)

    datasets_details: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {
        "n_days_train": float(df_train_g[p6.date_col].nunique()),
        "n_days_valid": float(df_valid_g[p6.date_col].nunique()),
        "n_days_test": float(df_test_g[p6.date_col].nunique()),
    }

    # 6) Por horizonte: raw windows + scaled windows + scaler
    artifacts_to_log: List[Path] = [report_path]

    for h in p6.horizons:
        cfg = map_by_h[h]
        feats = cfg["features"]
        tgt = cfg["target"]

        # raw npz paths
        raw_train_p = out_windows_dir / f"windows_train_{h}.npz"
        raw_valid_p = out_windows_dir / f"windows_valid_{h}.npz"
        raw_test_p = out_windows_dir / f"windows_test_{h}.npz"

        # scaled npz paths
        z_train_p = out_scaled_dir / f"windows_train_{h}_z.npz"
        z_valid_p = out_scaled_dir / f"windows_valid_{h}_z.npz"
        z_test_p = out_scaled_dir / f"windows_test_{h}_z.npz"
        scaler_p = out_scaled_dir / f"scaler_{h}.pkl"

        log.info("[6.%s] Construyendo/cargando windows RAW (%s)", h, tgt)
        Xtr, ytr = prepare_or_load_windows("train", df_train_g, feats, tgt, window_size, p6.date_col, raw_train_p)
        Xva, yva = prepare_or_load_windows("valid", df_valid_g, feats, tgt, window_size, p6.date_col, raw_valid_p)
        Xte, yte = prepare_or_load_windows("test", df_test_g, feats, tgt, window_size, p6.date_col, raw_test_p)

        log.info("[6.%s] Escalando (solo X) con fit en train", h)
        scale_and_save_windows(
            X_train=Xtr, X_valid=Xva, X_test=Xte,
            y_train=ytr, y_valid=yva, y_test=yte,
            scaler_type=scaler_type,
            scaler_path=scaler_p,
            out_train=z_train_p,
            out_valid=z_valid_p,
            out_test=z_test_p,
        )

        # checks/info (sobre scaled)
        info_tr = npz_info(z_train_p)
        info_va = npz_info(z_valid_p)
        info_te = npz_info(z_test_p)

        datasets_details[str(h)] = {
            "target": tgt,
            "n_features": int(Xtr.shape[2]) if Xtr.ndim == 3 else None,
            "feature_names": feats,
            "paths": {
                "raw": {"train": str(raw_train_p), "valid": str(raw_valid_p), "test": str(raw_test_p)},
                "scaled": {"train": str(z_train_p), "valid": str(z_valid_p), "test": str(z_test_p)},
                "scaler": str(scaler_p),
            },
            "scaled_info": {"train": info_tr, "valid": info_va, "test": info_te},
            "y_stats": {"train": flatten_stats(ytr), "valid": flatten_stats(yva), "test": flatten_stats(yte)},
        }

        artifacts_to_log += [raw_train_p, raw_valid_p, raw_test_p, z_train_p, z_valid_p, z_test_p, scaler_p]

        # métricas simples por horizonte
        metrics[f"h{h}_n_windows_train"] = float(info_tr["X_shape"][0])
        metrics[f"h{h}_n_windows_valid"] = float(info_va["X_shape"][0])
        metrics[f"h{h}_n_windows_test"] = float(info_te["X_shape"][0])

    details = {
        "config": {
            "window_size": int(window_size),
            "scaler_type": scaler_type,
            "scaler_scope": "train_only",
            "horizons": [int(x) for x in p6.horizons],
            "timezone_str": p6.timezone_str,
            "gestation_window_start": gw_start,
            "gestation_window_end": gw_end,
        },
        "datasets": datasets_details,
        "checks": {
            "no_leakage_assumed": True,
            "no_nan_after_scaling": all(
                (not d["scaled_info"]["train"]["has_nan_X"])
                and (not d["scaled_info"]["valid"]["has_nan_X"])
                and (not d["scaled_info"]["test"]["has_nan_X"])
                for d in datasets_details.values()
            ),
        },
    }

    paths = {
        "inputs": {
            "train_parquet": str(Path(args.in_train)),
            "valid_parquet": str(Path(args.in_valid)),
            "test_parquet": str(Path(args.in_test)),
            "stage04_summary": str(Path(args.in_stage04_summary)),
            "stage03b_summary": str(Path(args.in_stage03b_summary)),
        },
        "outputs": {
            "windows_dir": str(out_windows_dir),
            "scaled_dir": str(out_scaled_dir),
        },
        "reports": {"summary": str(report_path)},
    }

    params_used = {
        "date_col": p6.date_col,
        "horizons": [int(x) for x in p6.horizons],
        "scaler_type": scaler_type,
        "timezone_str": p6.timezone_str,
        "drop_feature_h60": p6.drop_feature_h60,
        "drop_feature_h90": p6.drop_feature_h90,
        "gestation_window_source": "stage_03b.summary.details.gestation_window",
    }

    envelope = build_stage06_envelope(
        created_at_utc=utc_now_iso(),
        version="1.0",
        paths=paths,
        params_used=params_used,
        metrics=metrics,
        details=details,
    )

    save_json(report_path, envelope)
    print_stage_summary_console(envelope)

    # MLflow
    mlflow_payload = build_mlflow_payload(envelope)
    log_to_mlflow(mlflow_payload, run_name="stage_06_window_scaling_seq2seq", artifacts=artifacts_to_log, enable=mlflow_enable)

    log.info("[DONE] stage_06_window_scaling_seq2seq finalizado correctamente.")


if __name__ == "__main__":
    main()
