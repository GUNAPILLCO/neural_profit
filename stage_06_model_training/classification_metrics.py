"""
classification_metrics.py

Utilidades de métricas para problemas de clasificación seq2one
aplicados al proyecto MNQ T2.

Diseñado para ser importado desde notebooks o scripts de entrenamiento.

Métricas principales:
- balanced_accuracy
- f1_macro
- f1_weighted

Métricas complementarias:
- accuracy
- precision_macro
- precision_weighted
- recall_macro
- recall_weighted

Diagnóstico:
- confusion_matrix
- class distribution real/predicha
- baseline naive (clase más frecuente en y_true)

Autor: OpenAI / Proyecto MNQ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetricsConfig:
    """
    Configuración para el cálculo de métricas de clasificación.
    """
    average_macro: str = "macro"
    average_weighted: str = "weighted"
    zero_division: int = 0
    include_confusion_matrix: bool = True
    include_class_distribution: bool = True
    include_naive_baseline: bool = True


def _to_1d_numpy(y: Iterable[Any], name: str) -> np.ndarray:
    """
    Convierte una entrada a np.ndarray 1D.

    Parámetros
    ----------
    y : iterable
        Etiquetas reales o predichas.
    name : str
        Nombre de la variable, solo para mensajes de error.

    Retorna
    -------
    np.ndarray
        Vector 1D.
    """
    arr = np.asarray(y)

    if arr.ndim == 0:
        raise ValueError(f"{name} no puede ser escalar; se esperaba un vector 1D.")
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise ValueError(f"{name} está vacío.")

    return arr


def _build_naive_predictions(y_true: np.ndarray) -> np.ndarray:
    """
    Construye un baseline naive que predice siempre la clase más frecuente en y_true.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    return np.full_like(y_true, fill_value=majority_class)


def _compute_core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    zero_division: int = 0,
) -> Dict[str, float]:
    """
    Calcula métricas principales y complementarias.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=zero_division)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=zero_division)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=zero_division)
        ),
    }


def compute_classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    *,
    model_name: Optional[str] = None,
    split: Optional[str] = None,
    target: Optional[str] = None,
    labels: Optional[Iterable[Any]] = None,
    config: Optional[ClassificationMetricsConfig] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas estandarizadas para clasificación seq2one.

    Parámetros
    ----------
    y_true : iterable
        Etiquetas reales.
    y_pred : iterable
        Etiquetas predichas.
    model_name : str, opcional
        Nombre del modelo.
    split : str, opcional
        Split evaluado: train / valid / test.
    target : str, opcional
        Nombre del target.
    labels : iterable, opcional
        Orden explícito de clases para la matriz de confusión.
        Si es None, se usa la unión ordenada de y_true e y_pred.
    config : ClassificationMetricsConfig, opcional
        Configuración del cálculo.

    Retorna
    -------
    dict
        Diccionario con:
        - metadata
        - métricas del modelo
        - baseline naive
        - matriz de confusión
        - distribuciones de clases
    """
    cfg = config or ClassificationMetricsConfig()

    y_true_arr = _to_1d_numpy(y_true, "y_true")
    y_pred_arr = _to_1d_numpy(y_pred, "y_pred")

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError(
            f"y_true e y_pred deben tener la misma longitud. "
            f"Recibido: {y_true_arr.shape[0]} vs {y_pred_arr.shape[0]}"
        )

    if labels is None:
        final_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    else:
        final_labels = np.asarray(list(labels))

    metrics = {
        "model": model_name,
        "split": split,
        "target": target,
        "n_samples": int(len(y_true_arr)),
        **_compute_core_metrics(
            y_true_arr,
            y_pred_arr,
            zero_division=cfg.zero_division,
        ),
    }

    if cfg.include_naive_baseline:
        y_pred_naive = _build_naive_predictions(y_true_arr)
        naive_metrics = _compute_core_metrics(
            y_true_arr,
            y_pred_naive,
            zero_division=cfg.zero_division,
        )
        metrics.update({
            "accuracy_naive": naive_metrics["accuracy"],
            "balanced_accuracy_naive": naive_metrics["balanced_accuracy"],
            "f1_macro_naive": naive_metrics["f1_macro"],
            "f1_weighted_naive": naive_metrics["f1_weighted"],
            "balanced_accuracy_gain_vs_naive": (
                metrics["balanced_accuracy"] - naive_metrics["balanced_accuracy"]
            ),
            "f1_macro_gain_vs_naive": (
                metrics["f1_macro"] - naive_metrics["f1_macro"]
            ),
            "f1_weighted_gain_vs_naive": (
                metrics["f1_weighted"] - naive_metrics["f1_weighted"]
            ),
        })

    result: Dict[str, Any] = {"metrics": metrics}

    if cfg.include_confusion_matrix:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=final_labels)
        result["confusion_matrix"] = cm.tolist()
        result["confusion_matrix_labels"] = final_labels.tolist()

    if cfg.include_class_distribution:
        true_classes, true_counts = np.unique(y_true_arr, return_counts=True)
        pred_classes, pred_counts = np.unique(y_pred_arr, return_counts=True)

        result["class_distribution_true"] = {
            str(cls): int(cnt) for cls, cnt in zip(true_classes, true_counts)
        }
        result["class_distribution_pred"] = {
            str(cls): int(cnt) for cls, cnt in zip(pred_classes, pred_counts)
        }

    return result


def metrics_to_flat_dict(metrics_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplana la salida de compute_classification_metrics para convertirla
    fácilmente en DataFrame.

    Parámetros
    ----------
    metrics_result : dict
        Salida de compute_classification_metrics.

    Retorna
    -------
    dict
        Diccionario plano solo con métricas y metadata.
    """
    if "metrics" not in metrics_result:
        raise ValueError("El diccionario recibido no contiene la clave 'metrics'.")

    return dict(metrics_result["metrics"])


def print_classification_report_block(metrics_result: Dict[str, Any]) -> None:
    """
    Imprime un bloque compacto y legible con las métricas principales.
    """
    m = metrics_result["metrics"]

    title = (
        f"CLASSIFICATION REPORT | "
        f"model={m.get('model')} | split={m.get('split')} | target={m.get('target')}"
    )
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    print(f"n_samples               : {m.get('n_samples')}")
    print(f"balanced_accuracy       : {m.get('balanced_accuracy'):.6f}")
    print(f"f1_macro                : {m.get('f1_macro'):.6f}")
    print(f"f1_weighted             : {m.get('f1_weighted'):.6f}")
    print(f"accuracy                : {m.get('accuracy'):.6f}")
    print(f"precision_macro         : {m.get('precision_macro'):.6f}")
    print(f"recall_macro            : {m.get('recall_macro'):.6f}")

    if "balanced_accuracy_naive" in m:
        print("-" * len(title))
        print(f"balanced_accuracy_naive : {m.get('balanced_accuracy_naive'):.6f}")
        print(f"f1_macro_naive          : {m.get('f1_macro_naive'):.6f}")
        print(f"f1_weighted_naive       : {m.get('f1_weighted_naive'):.6f}")
        print(f"bal_acc_gain_vs_naive   : {m.get('balanced_accuracy_gain_vs_naive'):.6f}")
        print(f"f1_macro_gain_vs_naive  : {m.get('f1_macro_gain_vs_naive'):.6f}")


__all__ = [
    "ClassificationMetricsConfig",
    "compute_classification_metrics",
    "metrics_to_flat_dict",
    "print_classification_report_block",
]