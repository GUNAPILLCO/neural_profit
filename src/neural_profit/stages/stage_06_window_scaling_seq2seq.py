# ============================================================
# stage_06_window_scaling_seq2seq.py
# STAGE_06 – WINDOW SCALING & FINAL DATASET ASSEMBLY (SEQ2SEQ MNQ)
#
# Lee:    data/splits/mnq_train.parquet
#         data/splits/mnq_valid.parquet
#         data/splits/mnq_test.parquet
# Lee:    reports/features_target_summary.json   (schema: features/targets)
#
# Guarda: data/windows/windows_{split}_{h}.npz
#         data/windows/scaled/windows_{split}_{h}_z.npz
#         data/windows/scaled/scaler_{h}.pkl
#
# Report: reports/window_scaled_summary.json
# ============================================================

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_06_window_scaling_seq2seq")

# ----------------------------
# MLflow (opcional)
# ----------------------------
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None

ENABLE_MLFLOW = bool(int(os.environ.get("ENABLE_MLFLOW", "0")))

# ============================================================
# Paths / IO (via env o defaults)
# ============================================================
IN_PARQUET_TRAIN = Path(os.environ.get("IN_PARQUET_TRAIN", "data/splits/mnq_train.parquet"))
IN_PARQUET_VALID = Path(os.environ.get("IN_PARQUET_VALID", "data/splits/mnq_valid.parquet"))
IN_PARQUET_TEST  = Path(os.environ.get("IN_PARQUET_TEST",  "data/splits/mnq_test.parquet"))
IN_ARTIFACT      = Path(os.environ.get("IN_ARTIFACT", "reports/features_target_summary.json"))

# Ventanas RAW (un archivo por split y horizonte)
OUT_WINDOWS_60_TRAIN = Path(os.environ.get("OUT_WINDOWS_60_TRAIN", "data/windows/windows_train_60.npz"))
OUT_WINDOWS_60_VALID = Path(os.environ.get("OUT_WINDOWS_60_VALID", "data/windows/windows_valid_60.npz"))
OUT_WINDOWS_60_TEST  = Path(os.environ.get("OUT_WINDOWS_60_TEST",  "data/windows/windows_test_60.npz"))

OUT_WINDOWS_90_TRAIN = Path(os.environ.get("OUT_WINDOWS_90_TRAIN", "data/windows/windows_train_90.npz"))
OUT_WINDOWS_90_VALID = Path(os.environ.get("OUT_WINDOWS_90_VALID", "data/windows/windows_valid_90.npz"))
OUT_WINDOWS_90_TEST  = Path(os.environ.get("OUT_WINDOWS_90_TEST",  "data/windows/windows_test_90.npz"))

# Ventanas escaladas
OUT_WINDOWS_60_TRAIN_Z = Path(os.environ.get("OUT_WINDOWS_60_TRAIN_Z", "data/windows/scaled/windows_train_60_z.npz"))
OUT_WINDOWS_60_VALID_Z = Path(os.environ.get("OUT_WINDOWS_60_VALID_Z", "data/windows/scaled/windows_valid_60_z.npz"))
OUT_WINDOWS_60_TEST_Z  = Path(os.environ.get("OUT_WINDOWS_60_TEST_Z",  "data/windows/scaled/windows_test_60_z.npz"))

OUT_WINDOWS_90_TRAIN_Z = Path(os.environ.get("OUT_WINDOWS_90_TRAIN_Z", "data/windows/scaled/windows_train_90_z.npz"))
OUT_WINDOWS_90_VALID_Z = Path(os.environ.get("OUT_WINDOWS_90_VALID_Z", "data/windows/scaled/windows_valid_90_z.npz"))
OUT_WINDOWS_90_TEST_Z  = Path(os.environ.get("OUT_WINDOWS_90_TEST_Z",  "data/windows/scaled/windows_test_90_z.npz"))

# Scalers (por horizonte)
OUT_SCALER_60 = Path(os.environ.get("OUT_SCALER_60", "data/windows/scaled/scaler_60.pkl"))
OUT_SCALER_90 = Path(os.environ.get("OUT_SCALER_90", "data/windows/scaled/scaler_90.pkl"))

OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/window_scaled_summary.json"))

# ============================================================
# Params loader (params.yaml)
# ============================================================
IN_PARAMS_YAML = Path(os.environ.get("IN_PARAMS_YAML", "params.yaml"))

def load_params(path: Path) -> Dict[str, Any]:
    """
    Carga params.yaml y devuelve un dict.
    Soporta estructura libre, pero asume que existe stage_06.gestation_window.{start,end}
    (o sus equivalentes) según el yaml.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró params.yaml: {path}")

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    return params


def get_gestation_window_from_params(params: Dict[str, Any]) -> tuple[str, str]:
    """
    Obtiene (start, end) desde params.yaml.
    Busca primero en:
      - params['stage_06']['gestation_window']['start'/'end']
    y si no existe, intenta alternativas razonables.
    """
    # Ruta recomendada
    try:
        gw = params["stage_06"]["gestation_window"]
        start = str(gw["start"])
        end = str(gw["end"])
        return start, end
    except Exception:
        pass

    # Alternativas (por si usted lo guardó distinto)
    candidates = [
        ("stage_06", "gestation_window_start", "gestation_window_end"),
        ("stage_06", "window_start", "window_end"),
        ("stage06", "gestation_window_start", "gestation_window_end"),
    ]

    for a, b, c in candidates:
        if a in params and b in params[a] and c in params[a]:
            return str(params[a][b]), str(params[a][c])

    raise KeyError(
        "No encontré la ventana de gestación en params.yaml. "
        "Formato recomendado:\n"
        "stage_06:\n"
        "  gestation_window:\n"
        "    start: '08:20'\n"
        "    end: '08:49'\n"
    )

##############################################33

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("[OK] JSON guardado: %s", path)


# ============================================================
# 1) Carga parquet
# ============================================================
def load_mnq_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")

    log.info("[OK] Cargando parquet: %s", path)
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    return df


# ============================================================
# 2) Carga de artifact features/targets
# ============================================================
def load_features_target(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el artifact: {path}")

    with open(path, "r", encoding="utf-8") as f:
        features_target_summary = json.load(f)

    features = features_target_summary["schema"]["features"]
    targets = features_target_summary["schema"]["targets"]

    # Mantengo su criterio (ajustar según su schema real):
    features_60 = [f for f in features if f != "roc_30"]
    features_90 = [f for f in features if f != "roc_60"]

    # Asumimos targets[0]=delta_pts_60, targets[1]=delta_pts_90
    target_60 = targets[0]
    target_90 = targets[1]

    return features_60, features_90, target_60, target_90


# ============================================================
# 3) Filtrado por ventana de gestación
# ============================================================
def filter_gestation_window(
    df: pd.DataFrame,
    start: str,
    end: str,
    date_col: str = "date",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El DataFrame debe tener un DatetimeIndex en df.index.")

    if date_col not in df.columns:
        raise KeyError(f"No se encontró la columna '{date_col}' en el DataFrame.")

    df = df.sort_index()

    idx_dates = df.index.date
    col_dates = pd.to_datetime(df[date_col]).dt.date
    if not (col_dates.values == idx_dates).all():
        raise ValueError(
            f"Inconsistencia entre '{date_col}' y la fecha del DatetimeIndex. "
            "Revise registros mal asignados."
        )

    out = df.between_time(start, end, inclusive="both").sort_index()

    out_idx_dates = out.index.date
    out_col_dates = pd.to_datetime(out[date_col]).dt.date
    if not (out_col_dates.values == out_idx_dates).all():
        raise ValueError(
            f"El filtrado generó inconsistencia entre '{date_col}' y el índice."
        )

    return out


# ============================================================
# 4) Determinar window_size
# ============================================================
def find_window_size(df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, date_col: str = "date") -> int:
    # Conteos por día (usamos median para evitar outliers)
    n_train = int(df_train.groupby(date_col).size().median())
    n_valid = int(df_valid.groupby(date_col).size().median())
    n_test  = int(df_test.groupby(date_col).size().median())

    if n_train == n_valid == n_test:
        window_size = n_train
        log.info("[OK] window_size (median rows/day) = %s", window_size)
        return window_size

    raise ValueError(f"Window size mismatch (median rows/day): train={n_train}, valid={n_valid}, test={n_test}")


# ============================================================
# 5) Ventanas seq2seq (UNA por día)
# ============================================================
def build_daily_seq2seq_windows(
    df: pd.DataFrame,
    features,
    target_col: str,
    window_size: int,
    date_col: str = "date",
):
    X, y = [], []

    for _, grupo in df.groupby(date_col):
        grupo = grupo.sort_index()

        if len(grupo) != window_size:
            continue

        X_day = grupo[features].values
        y_day = grupo[target_col].values

        if np.isnan(X_day).any() or np.isnan(y_day).any():
            continue

        X.append(X_day)
        y.append(y_day)

    return np.array(X), np.array(y)


def prepare_or_load_seq2seq_windows(
    mnq_train: pd.DataFrame,
    mnq_valid: pd.DataFrame,
    mnq_test: pd.DataFrame,
    features,
    target_col: str,
    window_size: int,
    out_windows_train: Path,
    out_windows_valid: Path,
    out_windows_test: Path,
    date_col: str = "date",
):
    def _load_or_build(split_name: str, df: pd.DataFrame, out_path: Path):
        _ensure_parent_dir(out_path)

        if not out_path.exists():
            log.info("No existe -> Generando %s (%s) y guardando en: %s", split_name, target_col, out_path)
            X, y = build_daily_seq2seq_windows(df, features, target_col, window_size, date_col=date_col)
            np.savez_compressed(out_path, X=X, y=y)
            log.info("Guardado: %s | X=%s | y=%s", out_path, X.shape, y.shape)
            return X, y

        log.info("Ya existe -> Cargando %s desde: %s", split_name, out_path)
        data = np.load(out_path)
        X, y = data["X"], data["y"]
        log.info("Cargado: %s | X=%s | y=%s", out_path, X.shape, y.shape)
        return X, y

    X_train, y_train = _load_or_build("train", mnq_train, out_windows_train)
    X_valid, y_valid = _load_or_build("valid", mnq_valid, out_windows_valid)
    X_test,  y_test  = _load_or_build("test",  mnq_test,  out_windows_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def xy_info_seq2seq(
    horizon_min: int,
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    n_features: int | None = None,
):
    log.info("Info X/y horizonte %s min:", horizon_min)

    for name, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        log.info("[%s] X shape=%s | y shape=%s", name, X.shape, y.shape)

        if X.ndim == 3:
            _, W, F = X.shape
            if n_features is not None and F != n_features:
                log.warning("[%s] n_features esperado=%s, encontrado=%s", name, n_features, F)
            if y.ndim == 2 and y.shape[1] != W:
                log.warning("[%s] window_size difiere: X.W=%s vs y.W=%s", name, W, y.shape[1])

        y_flat = np.asarray(y).ravel()
        log.info("[%s] y stats: mean=%.6f std=%.6f min=%.6f max=%.6f",
                 name, y_flat.mean(), y_flat.std(), y_flat.min(), y_flat.max())


# ============================================================
# 6) Escalado (solo X)
# ============================================================
def _choose_scaler(scaler_type="standard"):
    st = scaler_type.lower()
    if st in ["standard", "z", "zscore"]:
        return StandardScaler()
    if st in ["minmax", "min_max"]:
        return MinMaxScaler()
    raise ValueError("scaler_type debe ser 'standard' o 'minmax'")


def _fit_on_3d(X_train_3d, scaler):
    _, _, F = X_train_3d.shape
    scaler.fit(X_train_3d.reshape(-1, F))
    return scaler


def _transform_3d(X_3d, scaler):
    n, W, F = X_3d.shape
    Xf = X_3d.reshape(-1, F)
    return scaler.transform(Xf).reshape(n, W, F)


def scale_and_save(
    X_train,
    X_valid=None,
    X_test=None,
    scaler_type="standard",
    scaler_path="data/windows/scaled/global_scaler.pkl",
    verbose=True,
):
    scaler_path = Path(scaler_path)
    _ensure_parent_dir(scaler_path)

    scaler = _choose_scaler(scaler_type)
    scaler = _fit_on_3d(X_train, scaler)

    X_train_s = _transform_3d(X_train, scaler)
    X_valid_s = _transform_3d(X_valid, scaler) if X_valid is not None else None
    X_test_s  = _transform_3d(X_test,  scaler) if X_test  is not None else None

    joblib.dump(scaler, scaler_path)

    if verbose:
        log.info("Scaler guardado en: %s", scaler_path)
        log.info("Shapes escaladas: X_train=%s | X_valid=%s | X_test=%s",
                 X_train_s.shape,
                 None if X_valid_s is None else X_valid_s.shape,
                 None if X_test_s is None else X_test_s.shape)

    return X_train_s, X_valid_s, X_test_s, scaler


def scale_and_save_windows_seq2seq(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    out_train_npz: Path,
    out_valid_npz: Path,
    out_test_npz:  Path,
    out_scaler_path: Path,
    scaler_type: str = "standard",
    verbose: bool = True,
):
    _ensure_parent_dir(out_train_npz)
    _ensure_parent_dir(out_valid_npz)
    _ensure_parent_dir(out_test_npz)
    _ensure_parent_dir(out_scaler_path)

    X_train_s, X_valid_s, X_test_s, scaler = scale_and_save(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        scaler_type=scaler_type,
        scaler_path=str(out_scaler_path),
        verbose=verbose,
    )

    np.savez_compressed(out_train_npz, X=X_train_s, y=y_train)
    np.savez_compressed(out_valid_npz, X=X_valid_s, y=y_valid)
    np.savez_compressed(out_test_npz,  X=X_test_s,  y=y_test)

    if verbose:
        log.info("Windows escaladas guardadas:")
        log.info("  - train: %s", out_train_npz)
        log.info("  - valid: %s", out_valid_npz)
        log.info("  - test : %s", out_test_npz)

    return X_train_s, y_train, X_valid_s, y_valid, X_test_s, y_test, scaler


# ============================================================
# 7) Summary report stage_06 + pretty print
# ============================================================
def build_stage06_summary_report(
    report_path: Path,
    *,
    window_size: int,
    scaler_type: str,
    horizons: list,
    timezone_str: str = "America/New_York",
    time_window_str: str = "08:20–08:49",
    windows_paths_by_horizon: dict,
    scaler_paths_by_horizon: dict,
    feature_names_by_horizon: dict | None = None,
    include_scaler_stats: bool = True,
    include_y_stats: bool = True,
    verbose: bool = True,
):
    def _as_path(p):
        return p if isinstance(p, Path) else Path(str(p))

    def _load_npz_shapes_and_checks(npz_path: Path):
        data = np.load(npz_path)
        X = data["X"]
        y = data["y"]
        return {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "has_nan_X": bool(np.isnan(X).any()),
            "has_nan_y": bool(np.isnan(y).any()),
        }

    def _flatten_stats(arr: np.ndarray):
        a = np.asarray(arr).ravel()
        return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

    def _scaler_stats(scaler, feature_names=None):
        out = {"scaler_class": scaler.__class__.__name__}
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            out["type"] = "standard"
            means = scaler.mean_.tolist()
            stds = scaler.scale_.tolist()
            if feature_names and len(feature_names) == len(means):
                out["per_feature"] = {fn: {"mean": float(m), "std": float(s)} for fn, m, s in zip(feature_names, means, stds)}
            else:
                out["mean"] = [float(x) for x in means]
                out["std"] = [float(x) for x in stds]
        elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            out["type"] = "minmax"
            mins = scaler.data_min_.tolist()
            maxs = scaler.data_max_.tolist()
            if feature_names and len(feature_names) == len(mins):
                out["per_feature"] = {fn: {"min": float(mi), "max": float(ma)} for fn, mi, ma in zip(feature_names, mins, maxs)}
            else:
                out["min"] = [float(x) for x in mins]
                out["max"] = [float(x) for x in maxs]
        else:
            out["type"] = "unknown"
        return out

    report_path = _as_path(report_path)
    _ensure_parent_dir(report_path)

    report = {
        "stage": "stage_06_window_scaling_seq2seq",
        "description": "Escalado de ventanas SEQ2SEQ diarias (X) usando estadísticas SOLO de train. y se guarda sin escalar (delta_pts).",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "window_size": int(window_size),
            "scaler_type": str(scaler_type),
            "scaler_scope": "train_only",
            "horizons": [int(h) for h in horizons],
            "timezone": timezone_str,
            "time_window": time_window_str,
        },
        "datasets": {},
        "checks": {"no_leakage_assumed": True, "shape_consistency": True, "no_nan_after_scaling": True},
        "artifacts": {"windows_npz": {}, "scalers": {}},
        "notes": [
            "El target (y) se guarda sin escalar para mantener unidades en puntos (delta_pts).",
            "El scaler se ajusta únicamente con X_train (aplanando n_days×window_size) para evitar leakage.",
        ],
    }

    global_shape_ok = True
    global_nan_ok = True

    for h in horizons:
        h_key = str(int(h))
        feat_names = None
        if feature_names_by_horizon and h in feature_names_by_horizon:
            feat_names = list(feature_names_by_horizon[h])

        paths = windows_paths_by_horizon.get(h, {})
        train_p = _as_path(paths.get("train"))
        valid_p = _as_path(paths.get("valid"))
        test_p  = _as_path(paths.get("test"))

        report["artifacts"]["windows_npz"][h_key] = {"train": str(train_p), "valid": str(valid_p), "test": str(test_p)}

        info_train = _load_npz_shapes_and_checks(train_p)
        info_valid = _load_npz_shapes_and_checks(valid_p)
        info_test  = _load_npz_shapes_and_checks(test_p)

        def _is_expected(info):
            Xs, ys = info["X_shape"], info["y_shape"]
            ok = (len(Xs) == 3) and (len(ys) == 2) and (Xs[1] == window_size) and (ys[1] == window_size) and (Xs[0] == ys[0])
            return bool(ok)

        shape_ok = _is_expected(info_train) and _is_expected(info_valid) and _is_expected(info_test)
        nan_ok = (
            not info_train["has_nan_X"] and not info_train["has_nan_y"] and
            not info_valid["has_nan_X"] and not info_valid["has_nan_y"] and
            not info_test["has_nan_X"]  and not info_test["has_nan_y"]
        )

        global_shape_ok &= shape_ok
        global_nan_ok &= nan_ok

        n_features = info_train["X_shape"][2]

        report["datasets"][h_key] = {
            "n_features": int(n_features),
            "splits": {
                "train": int(info_train["X_shape"][0]),
                "valid": int(info_valid["X_shape"][0]),
                "test":  int(info_test["X_shape"][0]),
            },
            "shapes": {
                "train": {"X": info_train["X_shape"], "y": info_train["y_shape"]},
                "valid": {"X": info_valid["X_shape"], "y": info_valid["y_shape"]},
                "test":  {"X": info_test["X_shape"],  "y": info_test["y_shape"]},
            },
            "checks": {"shape_ok": shape_ok, "no_nan": nan_ok},
        }

        if include_y_stats:
            ytr = np.load(train_p)["y"]
            yva = np.load(valid_p)["y"]
            yte = np.load(test_p)["y"]
            report["datasets"][h_key]["y_stats"] = {"train": _flatten_stats(ytr), "valid": _flatten_stats(yva), "test": _flatten_stats(yte)}

        scaler_p = scaler_paths_by_horizon.get(h) or scaler_paths_by_horizon.get(h_key)
        if scaler_p is not None:
            scaler_p = _as_path(scaler_p)
            report["artifacts"]["scalers"][h_key] = str(scaler_p)
            if include_scaler_stats and scaler_p.exists():
                scaler_obj = joblib.load(scaler_p)
                report["datasets"][h_key]["scaler_stats"] = _scaler_stats(scaler_obj, feature_names=feat_names)
        else:
            report["artifacts"]["scalers"][h_key] = None

        if feat_names is not None:
            report["datasets"][h_key]["feature_names"] = feat_names

    report["checks"]["shape_consistency"] = bool(global_shape_ok)
    report["checks"]["no_nan_after_scaling"] = bool(global_nan_ok)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        log.info("Report stage_06 guardado en: %s", report_path)

    return report


def print_stage06_summary_pretty(summary: Dict[str, Any]) -> None:
    if not summary:
        print("Summary vacío (stage_06).")
        return

    config = summary.get("config", {})
    datasets = summary.get("datasets", {})
    checks = summary.get("checks", {})
    artifacts = summary.get("artifacts", {})

    print("\n" + "=" * 78)
    print("STAGE_06 – WINDOW SCALING & DATASET FINALIZATION (SEQ2SEQ MNQ)")
    print("=" * 78)

    print("\nConfiguración")
    print("-" * 78)
    print(f"Window size        : {config.get('window_size')}")
    print(f"Scaler type        : {config.get('scaler_type')}")
    print(f"Scaler scope       : {config.get('scaler_scope')}")
    print(f"Horizontes         : {config.get('horizons')}")
    print(f"Timezone           : {config.get('timezone')}")
    print(f"Ventana horaria    : {config.get('time_window')}")

    for h, info in datasets.items():
        print("\n" + "-" * 78)
        print(f"Horizonte {h} minutos")
        print("-" * 78)
        print(f"Features           : {info.get('n_features')}")

        splits = info.get("splits", {})
        print("Splits (n_days)")
        print(f"  Train            : {splits.get('train')}")
        print(f"  Valid            : {splits.get('valid')}")
        print(f"  Test             : {splits.get('test')}")

        shapes = info.get("shapes", {})
        if shapes:
            print("\nShapes")
            for split_name, sh in shapes.items():
                print(f"  {split_name:<6} -> X: {sh.get('X')} | y: {sh.get('y')}")

        h_checks = info.get("checks", {})
        if h_checks:
            print("\nChecks")
            print(f"  Shape OK         : {h_checks.get('shape_ok')}")
            print(f"  No NaN           : {h_checks.get('no_nan')}")

        y_stats = info.get("y_stats")
        if y_stats:
            print("\nTarget (y) stats – flatten")
            for split_name, st in y_stats.items():
                print(
                    f"  {split_name:<6} -> "
                    f"mean={st.get('mean'):.6f} | std={st.get('std'):.6f} | "
                    f"min={st.get('min'):.6f} | max={st.get('max'):.6f}"
                )

    print("\n" + "-" * 78)
    print("Checks globales")
    print("-" * 78)
    print(f"No leakage asumido : {checks.get('no_leakage_assumed')}")
    print(f"Shape consistente  : {checks.get('shape_consistency')}")
    print(f"No NaN post-scale  : {checks.get('no_nan_after_scaling')}")

    print("\n" + "-" * 78)
    print("Artifacts generados")
    print("-" * 78)
    win_art = artifacts.get("windows_npz", {})
    for h, paths in win_art.items():
        print(f"Horizonte {h}:")
        print(f"  Train  : {paths.get('train')}")
        print(f"  Valid  : {paths.get('valid')}")
        print(f"  Test   : {paths.get('test')}")

    scalers = artifacts.get("scalers", {})
    if scalers:
        print("\nScalers")
        for h, p in scalers.items():
            print(f"  {h:<6} : {p}")

    notes = summary.get("notes", [])
    if notes:
        print("\n" + "-" * 78)
        print("Notas")
        print("-" * 78)
        for n in notes:
            print(f"- {n}")

    print("\n" + "=" * 78)


# ============================================================
# 8) Main
# ============================================================
def main() -> None:
    # 0) Load params.yaml
    params = load_params(IN_PARAMS_YAML)
    gestation_window_start, gestation_window_end = get_gestation_window_from_params(params)

    
    log.info("[1] Cargando splits parquet")
    mnq_train = load_mnq_parquet(IN_PARQUET_TRAIN)
    mnq_valid = load_mnq_parquet(IN_PARQUET_VALID)
    mnq_test  = load_mnq_parquet(IN_PARQUET_TEST)

    log.info("[2] Cargando schema features/targets")
    features_60, features_90, target_60, target_90 = load_features_target(IN_ARTIFACT)

    log.info("[3] Filtrando ventana de gestación")
    gestation_window_start = "08:20"
    gestation_window_end   = "08:49"
    mnq_train_g = filter_gestation_window(mnq_train, gestation_window_start, gestation_window_end)
    mnq_valid_g = filter_gestation_window(mnq_valid, gestation_window_start, gestation_window_end)
    mnq_test_g  = filter_gestation_window(mnq_test,  gestation_window_start, gestation_window_end)

    log.info("[4] Calculando window_size")
    window_size = find_window_size(mnq_train_g, mnq_valid_g, mnq_test_g)

    log.info("[5] Construyendo ventanas RAW")
    X_train_60, y_train_60, X_valid_60, y_valid_60, X_test_60, y_test_60 = prepare_or_load_seq2seq_windows(
        mnq_train=mnq_train_g,
        mnq_valid=mnq_valid_g,
        mnq_test=mnq_test_g,
        features=features_60,
        target_col=target_60,
        window_size=window_size,
        out_windows_train=OUT_WINDOWS_60_TRAIN,
        out_windows_valid=OUT_WINDOWS_60_VALID,
        out_windows_test=OUT_WINDOWS_60_TEST,
        date_col="date",
    )
    xy_info_seq2seq(60, X_train_60, y_train_60, X_valid_60, y_valid_60, X_test_60, y_test_60, n_features=len(features_60))

    X_train_90, y_train_90, X_valid_90, y_valid_90, X_test_90, y_test_90 = prepare_or_load_seq2seq_windows(
        mnq_train=mnq_train_g,
        mnq_valid=mnq_valid_g,
        mnq_test=mnq_test_g,
        features=features_90,
        target_col=target_90,  # <- corregido
        window_size=window_size,
        out_windows_train=OUT_WINDOWS_90_TRAIN,
        out_windows_valid=OUT_WINDOWS_90_VALID,
        out_windows_test=OUT_WINDOWS_90_TEST,
        date_col="date",
    )
    xy_info_seq2seq(90, X_train_90, y_train_90, X_valid_90, y_valid_90, X_test_90, y_test_90, n_features=len(features_90))

    log.info("[6] Escalando ventanas (solo X) y guardando")
    scale_and_save_windows_seq2seq(
        X_train=X_train_60, y_train=y_train_60,
        X_valid=X_valid_60, y_valid=y_valid_60,
        X_test=X_test_60,   y_test=y_test_60,
        out_train_npz=OUT_WINDOWS_60_TRAIN_Z,
        out_valid_npz=OUT_WINDOWS_60_VALID_Z,
        out_test_npz=OUT_WINDOWS_60_TEST_Z,
        out_scaler_path=OUT_SCALER_60,
        scaler_type="standard",
    )

    scale_and_save_windows_seq2seq(
        X_train=X_train_90, y_train=y_train_90,
        X_valid=X_valid_90, y_valid=y_valid_90,
        X_test=X_test_90,   y_test=y_test_90,
        out_train_npz=OUT_WINDOWS_90_TRAIN_Z,
        out_valid_npz=OUT_WINDOWS_90_VALID_Z,
        out_test_npz=OUT_WINDOWS_90_TEST_Z,
        out_scaler_path=OUT_SCALER_90,
        scaler_type="standard",
    )

    windows_paths_by_horizon = {
        60: {"train": OUT_WINDOWS_60_TRAIN_Z, "valid": OUT_WINDOWS_60_VALID_Z, "test": OUT_WINDOWS_60_TEST_Z},
        90: {"train": OUT_WINDOWS_90_TRAIN_Z, "valid": OUT_WINDOWS_90_VALID_Z, "test": OUT_WINDOWS_90_TEST_Z},
    }
    scaler_paths_by_horizon = {60: OUT_SCALER_60, 90: OUT_SCALER_90}

    report = build_stage06_summary_report(
        report_path=OUT_SUMMARY,
        window_size=window_size,  # <- corregido
        scaler_type="standard",
        horizons=[60, 90],
        timezone_str="America/New_York",
        time_window_str=f"{gestation_window_start}–{gestation_window_end}",
        windows_paths_by_horizon=windows_paths_by_horizon,
        scaler_paths_by_horizon=scaler_paths_by_horizon,
        feature_names_by_horizon={60: features_60, 90: features_90},
        include_scaler_stats=True,
        include_y_stats=True,
        verbose=True,
    )

    print_stage06_summary_pretty(report)
    log.info("[DONE] Stage_06 finalizado correctamente.")


if __name__ == "__main__":
    main()
