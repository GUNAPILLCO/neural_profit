import numpy as np
from sklearn.metrics import r2_score

def compute_seq2seq_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_r2: bool = True,
) -> dict:
    """
    Calcula métricas simples y comparables para modelos seq2seq.

    Parámetros
    ----------
    y_true : np.ndarray
        Valores reales con shape (n_samples, seq_len)
        o (n_samples, seq_len, 1)
    y_pred : np.ndarray
        Valores predichos con shape (n_samples, seq_len)
        o (n_samples, seq_len, 1)
    compute_r2 : bool
        Indica si se debe calcular R² sobre la secuencia completa

    Retorna
    -------
    metrics : dict
        Diccionario con métricas globales y diagnóstico por paso
    """

    # ------------------------------------------------------------------
    # 1) Asegurar que las entradas sean arrays NumPy
    # ------------------------------------------------------------------

    # Convierte y_true a np.ndarray (por si viene como lista o tensor)
    y_true = np.asarray(y_true)

    # Convierte y_pred a np.ndarray
    y_pred = np.asarray(y_pred)

    # ------------------------------------------------------------------
    # 2) Normalizar dimensiones a (n_samples, seq_len)
    # ------------------------------------------------------------------

    # Si y_true tiene dimensión extra (…, 1), la elimina
    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    # Si y_pred tiene dimensión extra (…, 1), la elimina
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(-1)

    # Verifica que ambas matrices tengan exactamente el mismo shape
    assert y_true.shape == y_pred.shape, (
        "y_true y y_pred deben tener el mismo shape"
    )

    # Extrae número de muestras y longitud de la secuencia
    n_samples, seq_len = y_true.shape

    # ------------------------------------------------------------------
    # 3) Cálculo de errores
    # ------------------------------------------------------------------

    # Error firmado: diferencia entre predicción y valor real
    errors = y_pred - y_true

    # Error absoluto
    abs_errors = np.abs(errors)

    # ------------------------------------------------------------------
    # 4) Métricas globales (sobre toda la secuencia)
    # ------------------------------------------------------------------

    # MAE global: promedio del error absoluto en todos los pasos
    mae = abs_errors.mean()

    # RMSE global: raíz del promedio del error cuadrático
    rmse = np.sqrt((errors ** 2).mean())

    # ------------------------------------------------------------------
    # 5) Métrica direccional (último paso de la secuencia)
    # ------------------------------------------------------------------

    # Extrae el valor real del último paso (k = seq_len)
    y_true_last = y_true[:, -1]

    # Extrae el valor predicho del último paso
    y_pred_last = y_pred[:, -1]

    # Calcula la accuracy direccional:
    # compara si el signo del delta predicho coincide con el real
    da_last = np.mean(
        np.sign(y_true_last) == np.sign(y_pred_last)
    )

    # ------------------------------------------------------------------
    # 6) MAE por paso (solo diagnóstico)
    # ------------------------------------------------------------------

    # Calcula MAE para cada paso temporal k = 1..seq_len
    mae_per_step = abs_errors.mean(axis=0)  # shape: (seq_len,)

    # ------------------------------------------------------------------
    # 7) Construcción del diccionario de métricas
    # ------------------------------------------------------------------

    metrics = {
        # Error absoluto medio global
        "MAE": float(mae),

        # Raíz del error cuadrático medio global
        "RMSE": float(rmse),

        # Accuracy direccional en el último paso
        "DA_last": float(da_last),

        # MAE por paso (lista para serializar a JSON)
        "MAE_per_step": mae_per_step.tolist(),
    }

    # ------------------------------------------------------------------
    # 8) R² opcional (sobre toda la secuencia concatenada)
    # ------------------------------------------------------------------

    if compute_r2:
        metrics["R2"] = float(
            r2_score(
                y_true.flatten(),   # vectoriza la secuencia real
                y_pred.flatten(),   # vectoriza la secuencia predicha
            )
        )

    # ------------------------------------------------------------------
    # 9) Retorno final
    # ------------------------------------------------------------------

    return metrics