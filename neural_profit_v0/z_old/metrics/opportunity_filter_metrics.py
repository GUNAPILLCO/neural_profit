import numpy as np

def compute_opportunity_filter_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta_op: float,
    theta: float,
) -> dict:
    """
    Calcula métricas económicas simples para evaluar al modelo como
    filtro de oportunidad durante la ventana de gestación.

    No representa PnL real. Mide calidad de señal y capacidad de anticipar
    movimientos explotables.

    Parámetros
    ----------
    y_true : np.ndarray
        Delta real en puntos (n_samples, seq_len) o (n_samples, seq_len, 1)
    y_pred : np.ndarray
        Delta predicho en puntos (n_samples, seq_len) o (n_samples, seq_len, 1)
    delta_op : float
        Umbral económico que define un movimiento explotable
    theta : float
        Umbral mínimo de predicción para generar señal

    Retorna
    -------
    metrics : dict
        Métricas de filtro económico
    """

    # ------------------------------------------------------------
    # 1) Normalización de shapes
    # ------------------------------------------------------------
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(-1)

    assert y_true.shape == y_pred.shape, "Shapes incompatibles"

    # ------------------------------------------------------------
    # 2) Usamos el último paso como referencia económica
    #    (movimiento a h minutos)
    # ------------------------------------------------------------
    y_true_last = y_true[:, -1]
    y_pred_last = y_pred[:, -1]

    # ------------------------------------------------------------
    # 3) Definición de eventos reales (oportunidad económica)
    # ------------------------------------------------------------
    real_opportunity = np.abs(y_true_last) >= delta_op

    # ------------------------------------------------------------
    # 4) Definición de señales del modelo
    # ------------------------------------------------------------
    model_signal = np.abs(y_pred_last) >= theta

    # Señal correcta: magnitud suficiente y signo correcto
    correct_signal = (
        model_signal
        & real_opportunity
        & (np.sign(y_true_last) == np.sign(y_pred_last))
    )

    # ------------------------------------------------------------
    # 5) Métricas económicas como filtro
    # ------------------------------------------------------------

    # Recall: ¿cuántas oportunidades reales fueron detectadas?
    opportunity_recall = (
        correct_signal.sum() / real_opportunity.sum()
        if real_opportunity.sum() > 0 else 0.0
    )

    # Precision: ¿cuántas señales del modelo eran oportunidades reales?
    precision = (
        correct_signal.sum() / model_signal.sum()
        if model_signal.sum() > 0 else 0.0
    )

    # Coverage: qué proporción del tiempo el modelo marca oportunidad
    coverage = model_signal.mean()

    metrics = {
        "Opportunity_Recall": float(opportunity_recall),
        "Precision": float(precision),
        "Coverage": float(coverage),
    }

    return metrics