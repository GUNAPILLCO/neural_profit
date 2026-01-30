import numpy as np
from sklearn.metrics import r2_score

def compute_seq2one_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    compute_r2: bool = True,
    da_ignore_zeros: bool = True,
    allow_seq_inputs_take_last: bool = False,
) -> dict:
    """
    Calcula métricas simples y comparables para modelos seq2one.

    Parámetros
    ----------
    y_true : np.ndarray
        Valores reales con shape (n_samples,), (n_samples, 1)
        (opcional) (n_samples, seq_len) si allow_seq_inputs_take_last=True.
    y_pred : np.ndarray
        Valores predichos con shape (n_samples,), (n_samples, 1)
        (opcional) (n_samples, seq_len) si allow_seq_inputs_take_last=True.
    compute_r2 : bool
        Si True, calcula R² sobre el vector completo.
    da_ignore_zeros : bool
        Si True, ignora casos donde el signo sea 0 en y_true o y_pred al calcular DA.
    allow_seq_inputs_take_last : bool
        Si True, permite inputs 2D (n_samples, seq_len) y toma el último paso [:, -1].
        Útil si algún modelo devuelve secuencia pero usted lo evalúa como many-to-one.

    Retorna
    -------
    metrics : dict
        Diccionario con métricas globales.
    """

    # 1) Convertir a np.ndarray y forzar float
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 2) Normalizar dimensiones hacia (n_samples,)
    def _to_1d(y: np.ndarray, name: str) -> np.ndarray:
        if y.ndim == 1:
            return y
        if y.ndim == 2:
            # (n, 1) -> (n,)
            if y.shape[1] == 1:
                return y.squeeze(1)
            # (n, seq_len) -> tomar último si se permite
            if allow_seq_inputs_take_last:
                return y[:, -1]
            raise ValueError(
                f"{name} con shape {y.shape} no es válido para seq2one. "
                f"Se esperaba (n_samples,) o (n_samples, 1)."
            )
        if y.ndim == 3 and y.shape[-1] == 1:
            # (n, seq_len, 1) -> (n, seq_len) y luego último si se permite
            y2 = y.squeeze(-1)
            if allow_seq_inputs_take_last:
                return y2[:, -1]
            raise ValueError(
                f"{name} con shape {y.shape} parece seq2seq. "
                f"Active allow_seq_inputs_take_last=True si quiere tomar el último paso."
            )
        raise ValueError(
            f"{name}.ndim={y.ndim} no es válido. "
            f"Se esperaba (n,), (n,1) o (n,seq_len) si allow_seq_inputs_take_last=True."
        )

    y_true = _to_1d(y_true, "y_true")
    y_pred = _to_1d(y_pred, "y_pred")

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true y y_pred deben tener el mismo shape. "
            f"Recibido y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    n_samples = int(y_true.shape[0])
    if n_samples == 0:
        raise ValueError("y_true/y_pred no pueden estar vacíos.")

    # 3) Validación numérica básica
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Se encontraron NaN o inf en y_true/y_pred. "
                         "Limpie o enmascare antes de calcular métricas.")

    # 4) Errores
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # 5) Métricas principales
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt((errors ** 2).mean()))

    # 6) Métrica direccional (DA)
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    if da_ignore_zeros:
        mask = (sign_true != 0) & (sign_pred != 0)
        da = float(np.mean(sign_true[mask] == sign_pred[mask])) if mask.any() else float("nan")
        da_n = int(mask.sum())
    else:
        da = float(np.mean(sign_true == sign_pred))
        da_n = n_samples

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "DA": da,
        "DA_n": da_n,  # cuántas muestras realmente aportaron a DA (si ignore_zeros=True)
    }

    # 7) R² opcional
    if compute_r2:
        metrics["R2"] = float(r2_score(y_true, y_pred))

    return metrics
