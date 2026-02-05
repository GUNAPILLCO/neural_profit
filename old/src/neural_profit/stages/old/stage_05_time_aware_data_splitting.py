# ============================================================
# stage_05_time_aware_data_splitting.py
# Time-aware split (MNQ intraday) por jornadas: Train / Valid / Test
# Lee:    data/features/mnq_features_target.parquet
# Lee:    reports/features_target_summary.json   (schema: features/targets)
# Guarda: data/splits/mnq_train.parquet
#         data/splits/mnq_valid.parquet
#         data/splits/mnq_test.parquet
#         data/splits/splits.json
# Report: reports/splits_summary.json
# ============================================================

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import yaml

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("stage_05_time_aware_data_splitting")

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
IN_PARQUET = Path(os.environ.get("IN_PARQUET", "data/features/mnq_features_target.parquet"))
IN_ARTIFACT = Path(os.environ.get("IN_ARTIFACT", "reports/features_target_summary.json"))

OUT_SPLITS = Path(os.environ.get("OUT_SPLITS", "data/splits/splits.json"))
OUT_PARQUET_TRAIN = Path(os.environ.get("OUT_PARQUET_TRAIN", "data/splits/mnq_train.parquet"))
OUT_PARQUET_VALID = Path(os.environ.get("OUT_PARQUET_VALID", "data/splits/mnq_valid.parquet"))
OUT_PARQUET_TEST = Path(os.environ.get("OUT_PARQUET_TEST", "data/splits/mnq_test.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/splits_summary.json"))

# Split ratios (por días)
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.70"))
VALID_RATIO = float(os.environ.get("VALID_RATIO", "0.15"))

# Validación de gaps
EXPECTED_GAP_MINUTES = int(os.environ.get("EXPECTED_GAP_MINUTES", "1"))

# MLflow enable flag
ENABLE_MLFLOW = bool(int(os.environ.get("ENABLE_MLFLOW", "0")))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info("[OK] JSON guardado: %s", path)


# ============================================================
# 1) Carga y preprocesamiento base
# ============================================================
def load_mnq_parquet(path: Path = IN_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el parquet de entrada: {path}")
    log.info("[OK] Cargando parquet: %s", path)
    return pd.read_parquet(path)


def add_column_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Asegura DatetimeIndex y agrega columna 'date' (YYYY-MM-DD) para agrupar por jornada.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out[date_col] = out.index.date

    # Reordenar: date primero
    cols = [date_col] + [c for c in out.columns if c != date_col]
    return out[cols]


# ============================================================
# 2) Carga de listado de features y targets (artifact JSON)
# ============================================================
def load_features_targets_list(path: Path = IN_ARTIFACT) -> Tuple[List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el artifact de features/targets: {path}")

    log.info("[2] Cargando features/targets desde: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        features_target_summary = json.load(f)

    features = features_target_summary["schema"]["features"]
    targets = features_target_summary["schema"]["targets"]

    log.info("[OK] Features=%d | Targets=%d", len(features), len(targets))
    return features, targets


# ============================================================
# 3) Validación dura de NaNs (corta ejecución si existen)
# ============================================================
def nan_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica la existencia de NaN por día y por columna.
    Si detecta algún NaN, loguea el detalle y corta la ejecución del script.
    """
    if "date" not in df.columns:
        raise ValueError("La columna 'date' no existe. Ejecute add_column_date() antes de validar NaNs.")

    log.info("[3.1] Iniciando validación de NaN por día y por columna")

    daily_nan_counts = df.groupby("date").apply(lambda x: x.isna().sum())
    nan_mask = daily_nan_counts > 0

    if nan_mask.any().any():
        log.info("[ERROR] Se detectaron valores NaN en el dataset")

        nan_details = (
            daily_nan_counts[nan_mask]
            .stack()
            .reset_index()
            .rename(columns={"level_1": "column", 0: "nan_count"})
        )

        log.info("[ERROR] Detalle de NaN encontrados (fecha, columna, cantidad):")
        log.info("\n%s", nan_details.to_string(index=False))

        raise RuntimeError("[ERROR] Ejecución detenida por presencia de NaN en el dataset")

    log.info("[OK] Validación NaN OK: no se encontraron valores faltantes")

    daily_unique_nans = pd.DataFrame(
        {
            "feature": daily_nan_counts.columns,
            "daily_nan_counts": [sorted(daily_nan_counts[col].unique()) for col in daily_nan_counts.columns],
        }
    )
    return daily_unique_nans


# ============================================================
# 4) Validación dura de gaps (corta ejecución si existen)
# ============================================================
def detectar_gaps(df: pd.DataFrame, gap_minutes: int = 1) -> None:
    """
    Verifica si existen gaps distintos al intervalo esperado (por defecto 1 minuto)
    entre registros consecutivos dentro de cada día.

    Si se detectan gaps:
      - Se loguea el detalle
      - Se lanza una excepción y se corta la ejecución

    Si no se detectan gaps:
      - Continúa normalmente
    """
    log.info("[3.2] Iniciando validación de gaps temporales")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El índice del DataFrame debe ser DatetimeIndex para detectar gaps.")

    work = df.copy()
    work["time_diff"] = work.index.to_series().diff()

    base_time_diff = pd.Timedelta(minutes=gap_minutes)
    gap_events: List[Dict[str, Any]] = []

    for day, group in work.groupby(work.index.date):
        time_diff = group["time_diff"].iloc[1:]  # omitir primer registro del día
        gaps = time_diff[time_diff != base_time_diff]

        if not gaps.empty:
            for idx, diff in gaps.items():
                gap_events.append({"date": str(day), "timestamp": str(idx), "time_diff": str(diff)})

    if gap_events:
        log.info("[ERROR] Se detectaron gaps temporales en el dataset")

        gap_df = pd.DataFrame(gap_events)
        log.info("[ERROR] Detalle de gaps detectados (fecha, timestamp, diferencia):")
        log.info("\n%s", gap_df.to_string(index=False))

        raise RuntimeError("[ERROR] Ejecución detenida por detección de gaps temporales en el dataset")

    log.info("[OK] Validación de gaps OK: todas las muestras son consecutivas cada %d minuto(s)", gap_minutes)


# ============================================================
# 5) CARGA DE RATIOS DESDE PARAMS.YAML
# ============================================================

PARAMS_YAML = Path(os.environ.get("PARAMS_YAML", "params.yaml"))

def load_stage05_params(params_path: Path = PARAMS_YAML) -> Tuple[float, float, int]:
    if not params_path.exists():
        raise FileNotFoundError(f"No se encontró params.yaml: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        p = yaml.safe_load(f) or {}

    s5 = (p.get("stage_05") or {})
    train_ratio = float(s5.get("train_ratio", 0.70))
    valid_ratio = float(s5.get("valid_ratio", 0.15))
    expected_gap_minutes = int(s5.get("expected_gap_minutes", 1))

    return train_ratio, valid_ratio, expected_gap_minutes

# ============================================================
# 5) Summary JSON + Pretty logs (stage_05)
# ============================================================
def build_splits_json(train_days: pd.Index, val_days: pd.Index, test_days: pd.Index) -> Dict[str, Any]:
    return {
        "train_days": [str(d) for d in train_days],
        "valid_days": [str(d) for d in val_days],
        "test_days": [str(d) for d in test_days],
        "n_train_days": int(len(train_days)),
        "n_valid_days": int(len(val_days)),
        "n_test_days": int(len(test_days)),
    }


def build_splits_summary_report(
    df_in: pd.DataFrame,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    targets: List[str],
    date_col: str = "date",
    decimals: int = 6,
) -> Dict[str, Any]:

    def _stats(df: pd.DataFrame) -> Dict[str, Any]:
        n_rows = int(df.shape[0])
        n_days = int(df[date_col].nunique()) if (date_col in df.columns and n_rows > 0) else 0
        rows_per_day = float(df.groupby(date_col).size().mean()) if (date_col in df.columns and n_rows > 0) else None
        dt_min = str(df.index.min()) if n_rows > 0 else None
        dt_max = str(df.index.max()) if n_rows > 0 else None
        return {
            "n_rows": n_rows,
            "n_days": n_days,
            "rows_per_day_mean": None if rows_per_day is None else round(rows_per_day, 2),
            "datetime_min": dt_min,
            "datetime_max": dt_max,
        }

    final_cols = [date_col] + features + targets

    def _nan_ratio_map(df: pd.DataFrame) -> Dict[str, float]:
        cols_existing = [c for c in final_cols if c in df.columns]
        out = {c: float(df[c].isna().mean()) for c in cols_existing}
        return {k: round(v, decimals) for k, v in out.items()}

    summary = {
        "stage": "stage_05_time_aware_data_splitting",
        "description": "Split temporal por jornadas (train/valid/test) sin mezcla entre días.",
        "io": {
            "in_parquet": str(IN_PARQUET),
            "in_artifact_features_targets": str(IN_ARTIFACT),
            "out_train": str(OUT_PARQUET_TRAIN),
            "out_valid": str(OUT_PARQUET_VALID),
            "out_test": str(OUT_PARQUET_TEST),
            "out_splits_json": str(OUT_SPLITS),
            "out_summary": str(OUT_SUMMARY),
        },
        "schema": {
            "date_col": date_col,
            "features": features,
            "targets": targets,
            "n_features": int(len(features)),
            "n_targets": int(len(targets)),
        },
        "ratios": {
            "train_ratio": TRAIN_RATIO,
            "valid_ratio": VALID_RATIO,
            "test_ratio": round(1.0 - TRAIN_RATIO - VALID_RATIO, 6),
        },
        "splits": {
            "input": _stats(df_in),
            "train": _stats(df_train),
            "valid": _stats(df_valid),
            "test": _stats(df_test),
        },
        "nan_ratio_by_col": {
            "train": _nan_ratio_map(df_train),
            "valid": _nan_ratio_map(df_valid),
            "test": _nan_ratio_map(df_test),
        },
    }

    return summary


def print_splits_summary_pretty(summary: Dict[str, Any]) -> None:
    if not summary:
        log.info("Summary vacío (stage_05).")
        return

    schema = summary.get("schema", {})
    splits = summary.get("splits", {})
    ratios = summary.get("ratios", {})

    log.info("============================================================")
    log.info("STAGE_05 – TIME AWARE DATA SPLITTING (MNQ)")
    log.info("============================================================")
    log.info("Ratios: train=%s | valid=%s | test=%s",
             ratios.get("train_ratio"), ratios.get("valid_ratio"), ratios.get("test_ratio"))
    log.info("Features finales (%s): %s", schema.get("n_features", "?"), schema.get("features", []))
    log.info("Targets finales  (%s): %s", schema.get("n_targets", "?"), schema.get("targets", []))

    for k in ["input", "train", "valid", "test"]:
        s = splits.get(k, {})
        log.info("------------------------------------------------------------")
        log.info("%s", k.upper())
        log.info("rows=%s | days=%s | rows/day=%s | dt=[%s .. %s]",
                 s.get("n_rows"), s.get("n_days"), s.get("rows_per_day_mean"),
                 s.get("datetime_min"), s.get("datetime_max"))

    log.info("============================================================")


# ============================================================
# 6) MLflow helpers (reutilizable)
# ============================================================
def build_stage05_mlflow_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    schema = summary.get("schema", {})
    splits = summary.get("splits", {})

    train = splits.get("train", {})
    valid = splits.get("valid", {})
    test = splits.get("test", {})

    out = {
        "params": {
            "stage": summary.get("stage", "stage_05_time_aware_data_splitting"),
            "n_features": int(schema.get("n_features", 0)),
            "n_targets": int(schema.get("n_targets", 0)),
            "train_ratio": float(summary.get("ratios", {}).get("train_ratio", np.nan)),
            "valid_ratio": float(summary.get("ratios", {}).get("valid_ratio", np.nan)),
        },
        "metrics": {
            "train_rows": float(train.get("n_rows", np.nan)),
            "valid_rows": float(valid.get("n_rows", np.nan)),
            "test_rows": float(test.get("n_rows", np.nan)),
            "train_days": float(train.get("n_days", np.nan)),
            "valid_days": float(valid.get("n_days", np.nan)),
            "test_days": float(test.get("n_days", np.nan)),
        },
    }
    return out


def log_mlflow(summary_mlflow: Dict[str, Any], run_name: str, artifacts: List[str] | None = None, enable: bool = True) -> None:
    if (not enable) or (mlflow is None):
        log.info("[MLflow] Deshabilitado (ENABLE_MLFLOW=%s) o mlflow no disponible.", enable)
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
                try:
                    if p and os.path.exists(p):
                        mlflow.log_artifact(p)
                except Exception:
                    pass

    log.info("[MLflow] Run registrado: %s", run_name)


# ============================================================
# 7) Main
# ============================================================
def main() -> None:
    # 1) Load
    log.info("[1] Cargando mnq_features_target.parquet")
    df_in = load_mnq_parquet(IN_PARQUET)
    df_in = add_column_date(df_in, date_col="date").sort_index()

    # 2) Load features y targets
    features, targets = load_features_targets_list(IN_ARTIFACT)

    # 3) Validaciones duras
    train_ratio, valid_ratio, expected_gap_minutes = load_stage05_params()

    
    log.info("[3] Validaciones (NaNs + gaps)")
    nan_count(df_in)
    detectar_gaps(df_in, gap_minutes=expected_gap_minutes)

    # 4) Split por días
       
    log.info("[4] Generando split temporal por días")
    unique_days = pd.Index(sorted(df_in["date"].unique()))
    num_dias = len(unique_days)



    if num_dias <= 0:
        raise RuntimeError("[ERROR] No se encontraron días en el dataset (columna 'date' vacía).")

    if not (0.0 < train_ratio < 1.0) or not (0.0 < valid_ratio < 1.0) or (train_ratio + valid_ratio >= 1.0):
        raise ValueError("[ERROR] train_ratio y valid_ratio inválidos. Requiere: 0<train<1, 0<valid<1, train+valid<1.")

    
    
    n_train = int(num_dias * train_ratio)
    n_valid = int(num_dias * valid_ratio)
    n_test = num_dias - n_train - n_valid

    # Seguridad mínima
    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(
            f"[ERROR] Split inválido por cantidad de días. total={num_dias}, train={n_train}, valid={n_valid}, test={n_test}."
        )

    log.info("[OK] Días totales=%d | train=%d | valid=%d | test=%d", num_dias, n_train, n_valid, n_test)

    train_days = unique_days[:n_train]
    val_days = unique_days[n_train:n_train + n_valid]
    test_days = unique_days[n_train + n_valid:]

    # 5) Construir datasets
    log.info("[5] Construyendo datasets train/valid/test")
    df_train = df_in[df_in["date"].isin(train_days)].copy().sort_index()
    df_valid = df_in[df_in["date"].isin(val_days)].copy().sort_index()
    df_test = df_in[df_in["date"].isin(test_days)].copy().sort_index()

    # 6) Guardar outputs parquet + splits.json
    log.info("[6] Guardando outputs parquet + splits.json")
    _ensure_parent_dir(OUT_PARQUET_TRAIN)
    _ensure_parent_dir(OUT_PARQUET_VALID)
    _ensure_parent_dir(OUT_PARQUET_TEST)

    df_train.to_parquet(OUT_PARQUET_TRAIN, index=True)
    df_valid.to_parquet(OUT_PARQUET_VALID, index=True)
    df_test.to_parquet(OUT_PARQUET_TEST, index=True)

    log.info("[OK] Guardado train: %s", OUT_PARQUET_TRAIN)
    log.info("[OK] Guardado valid: %s", OUT_PARQUET_VALID)
    log.info("[OK] Guardado test : %s", OUT_PARQUET_TEST)

    splits_payload = build_splits_json(train_days, val_days, test_days)
    save_json(OUT_SPLITS, splits_payload)

    # 7) Summary JSON + pretty logs
    log.info("[7] Generando summary (JSON) y pretty logs")
    summary = build_splits_summary_report(
        df_in=df_in,
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        features=features,
        targets=targets,
        date_col="date",
    )
    save_json(OUT_SUMMARY, summary)
    print_splits_summary_pretty(summary)

    # 8) MLflow (opcional)
    log.info("[8] MLflow (opcional)")
    mlflow_summary = build_stage05_mlflow_summary(summary)
    log_mlflow(
        summary_mlflow=mlflow_summary,
        run_name="stage_05_time_aware_data_splitting",
        artifacts=[str(OUT_SUMMARY), str(OUT_SPLITS)],
        enable=ENABLE_MLFLOW,
    )

    log.info("[DONE] Stage_05 finalizado correctamente.")


if __name__ == "__main__":
    main()
