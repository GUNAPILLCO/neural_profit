

IN_PARQUET = Path(os.environ.get("IN_PARQUET", "data/processed/mnq_intraday.parquet"))
OUT_SUMMARY = Path(os.environ.get("OUT_SUMMARY", "reports/target_investigation_summary.json"))

def load_mnq_parquet():
    os.path.exists(IN_PARQUET)
    print("Archivo encontrado en disco. Cargando dataset local...")
    mnq_parquet = pd.read_parquet(IN_PARQUET)
    return mnq_parquet

def add_column_date(df):
    # Asegurar que el índice esté en formato datetime
    df.index = pd.to_datetime(df.index)

    # Crear una nueva columna 'date' con la fecha extraída del índice
    df['date'] = df.index.date

    # Reordenar columnas: 'date', 'time_str', y luego el resto
    cols = ['date'] + [col for col in df.columns if col not in ['date']]

    df = df[cols]

    return df


def add_future_delta_pts_by_day(
    df: pd.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    horizons=(60, 90),
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(date_col, sort=False)

    for h in horizons:
        fut_close = g[close_col].shift(-h)                 # close_{t+h} dentro del mismo día
        out[f"delta_pts_{h}"] = fut_close - out[close_col] # Δpts_{t,h}
        out[f"abs_delta_pts_{h}"] = out[f"delta_pts_{h}"].abs()
        out[f"sign_delta_{h}"] = np.sign(out[f"delta_pts_{h}"]).astype("float")

    return out

def add_minute_of_day(df):
  df["minute_of_day"] = df.index.hour * 60 + df.index.minute
  df["hour"] = df.index.hour
  df["minute"] = df.index.minute
  return df


def stats_by_minute_of_day(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Calcula estadísticos por minute_of_day para un horizonte h:
      - n
      - mediana y percentiles de |Δ|
      - media de |Δ|
      - proporción de Δ>0 y Δ<0
    Requiere columnas:
      - delta_pts_{h}
      - abs_delta_pts_{h}
      - minute_of_day
    """
    base = df[["minute_of_day", f"delta_pts_{h}", f"abs_delta_pts_{h}"]].dropna()

    g = base.groupby("minute_of_day", sort=True)

    out = g.agg(
        n=(f"abs_delta_pts_{h}", "size"),
        mean_abs=(f"abs_delta_pts_{h}", "mean"),
        median_abs=(f"abs_delta_pts_{h}", "median"),
    )

    # Percentiles (robustos y útiles para ver colas por horario)
    for q in [0.60, 0.70, 0.80, 0.90]:
        out[f"p{int(q*100)}_abs"] = g[f"abs_delta_pts_{h}"].quantile(q)

    # Sesgo direccional simple (sin umbrales)
    out["pos_ratio"] = g[f"delta_pts_{h}"].apply(lambda x: (x > 0).mean())
    out["neg_ratio"] = g[f"delta_pts_{h}"].apply(lambda x: (x < 0).mean())

    # Hora/minuto para lectura
    out["hour"] = (out.index // 60).astype(int)
    out["minute"] = (out.index % 60).astype(int)

    return out.sort_index()



def top_minutes(
    df_by_time: pd.DataFrame,
    col: str,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Devuelve el top horarios según una métrica (col) y
    muestra un resumen con:
      - hora mínima y máxima del top
      - valor mínimo y máximo de la métrica
    """

    cols_show = [
        "hour", "minute", "n",
        "mean_abs", "median_abs",
        "p60_abs", "p70_abs", "p80_abs", "p90_abs",
        "pos_ratio", "neg_ratio",
    ]

    top_df = (
        df_by_time
        .sort_values(col, ascending=False)
        [cols_show]
        .head(top_n)
        .copy()
    )

    # Construcción HH:MM
    top_df["minute_of_day"] = top_df["hour"] * 60 + top_df["minute"]

    # Resumen horario
    min_mod = top_df["minute_of_day"].min()
    max_mod = top_df["minute_of_day"].max()

    min_time = f"{min_mod//60:02d}:{min_mod%60:02d}"
    max_time = f"{max_mod//60:02d}:{max_mod%60:02d}"

    min_val = top_df[col].min()
    max_val = top_df[col].max()

    print("=" * 60)
    print(f"Top {top_n} según '{col}'")
    print(f"Ventana horaria: {min_time}  →  {max_time}")
    print(f"{col} mínimo: {min_val:.2f}")
    print(f"{col} máximo: {max_val:.2f}")
    print("=" * 60)

    return top_df.drop(columns="minute_of_day")


import pandas as pd
from typing import Tuple

def delta_midpoint(delta_min: float, delta_max: float) -> float:
    """
    Calcula el punto medio entre dos valores de delta.
    """
    return delta_min + (delta_max - delta_min) / 2


def optimal_window_from_rankings(*tops: pd.DataFrame):
    """
    Recibe varios DataFrames de rankings (top_med, top_p70, top_p90)
    y devuelve la ventana óptima como intersección horaria [HH:MM, HH:MM].
    """

    # Pasamos todo a minute_of_day para comparar correctamente
    mins_start = []
    mins_end = []

    for df in tops:
        minute_of_day = df["hour"] * 60 + df["minute"]
        mins_start.append(minute_of_day.min())
        mins_end.append(minute_of_day.max())

    # Intersección robusta
    start_opt = max(mins_start)
    end_opt = min(mins_end)

    if start_opt > end_opt:
        raise ValueError("No hay intersección horaria entre los rankings.")

    # Formato HH:MM
    start_hhmm = f"{start_opt // 60:02d}:{start_opt % 60:02d}"
    end_hhmm = f"{end_opt // 60:02d}:{end_opt % 60:02d}"

    return {
        "start_minute_of_day": start_opt,
        "end_minute_of_day": end_opt,
        "start_hhmm": start_hhmm,
        "end_hhmm": end_hhmm,
    }

###############

#STOP LOSS

@njit
def _mae_until_tp_numba(close, high, low, sign, h, tp_pts):
    """
    Calcula MAE hasta TP o hasta horizonte h (lo que ocurra primero).
    Implementación rápida (Numba) para un solo día.
    """
    n = close.shape[0]
    mae = np.full(n, np.nan)
    tp_hit = np.zeros(n, dtype=np.bool_)
    tau = np.full(n, np.nan)

    for i in range(n):
        # Si no hay h minutos hacia adelante, no se puede calcular
        if i + h >= n:
            continue

        s = sign[i]
        if not np.isfinite(s) or s == 0.0:
            continue

        entry = close[i]
        u_exit = h  # por defecto: vence el horizonte

        if s > 0:  # LONG
            tp_price = entry + tp_pts

            # Buscar primer u (1..h) donde high[i+u] >= tp_price
            for u in range(1, h + 1):
                if high[i + u] >= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

            # MAE LONG: entry - min(low) en [i+1, i+u_exit]
            min_low = low[i + 1]
            for u in range(2, u_exit + 1):
                v = low[i + u]
                if v < min_low:
                    min_low = v

            mae[i] = entry - min_low
            tau[i] = u_exit

        else:      # SHORT
            tp_price = entry - tp_pts

            # Buscar primer u (1..h) donde low[i+u] <= tp_price
            for u in range(1, h + 1):
                if low[i + u] <= tp_price:
                    u_exit = u
                    tp_hit[i] = True
                    break

            # MAE SHORT: max(high) en [i+1, i+u_exit] - entry
            max_high = high[i + 1]
            for u in range(2, u_exit + 1):
                v = high[i + u]
                if v > max_high:
                    max_high = v

            mae[i] = max_high - entry
            tau[i] = u_exit

    return mae, tp_hit, tau


def compute_mae_until_tp_fast(
    df: pd.DataFrame,
    h: int,
    tp_pts: float,
    date_col: str = "date",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    sign_col: str | None = None,  # por defecto: sign_delta_{h}
) -> pd.DataFrame:
    """
    Wrapper por día (evita cruzar sesiones) + Numba para velocidad.
    Devuelve columnas:
      - mae_tp_{h}
      - tp_hit_{h}
      - tau_{h}
    """
    if sign_col is None:
        sign_col = f"sign_delta_{h}"

    required = {date_col, close_col, high_col, low_col, sign_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    mae_out = np.full(len(df), np.nan, dtype=float)
    hit_out = np.zeros(len(df), dtype=bool)
    tau_out = np.full(len(df), np.nan, dtype=float)

    # Trabajamos por bloques de día. Importante: conservar orden del df.
    # Para asignar rápido, usamos posiciones (iloc) del bloque.
    for _, g in df.groupby(date_col, sort=False):
        pos = g.index.to_numpy()
        # Convertimos a posiciones enteras en el df (para asignar en arrays)
        # Si el índice no es RangeIndex, usamos get_indexer (una vez por día).
        locs = df.index.get_indexer(pos)

        close = g[close_col].to_numpy(dtype=np.float64)
        high = g[high_col].to_numpy(dtype=np.float64)
        low  = g[low_col].to_numpy(dtype=np.float64)
        sign = g[sign_col].to_numpy(dtype=np.float64)

        mae_d, hit_d, tau_d = _mae_until_tp_numba(close, high, low, sign, h, tp_pts)

        mae_out[locs] = mae_d
        hit_out[locs] = hit_d
        tau_out[locs] = tau_d

    return pd.DataFrame(
        {
            f"mae_tp_{h}": mae_out,
            f"tp_hit_{h}": hit_out,
            f"tau_{h}": tau_out,
        },
        index=df.index
    )



def mae_percentiles_by_time(
    df: pd.DataFrame,
    mae_col: str,
    percentiles=(0.7, 0.8),
    minute_col: str = "minute_of_day",
) -> pd.DataFrame:
    """
    Calcula percentiles del MAE por minute_of_day.

    Parámetros
    ----------
    df : DataFrame
        Dataset que contiene mae_tp_h y minute_of_day.
    mae_col : str
        Nombre de la columna MAE (ej: 'mae_tp_60').
    percentiles : tuple
        Percentiles a calcular (ej: (0.7, 0.8)).
    """

    base = df[[minute_col, mae_col]].dropna().copy()
    g = base.groupby(minute_col)[mae_col]

    out = pd.DataFrame(index=g.size().index)
    out["n"] = g.size()

    for p in percentiles:
        out[f"p{int(p*100)}"] = g.quantile(p)

    # Reconstrucción HH:MM
    out = out.reset_index()
    out["hour"] = (out[minute_col] // 60).astype(int)
    out["minute"] = (out[minute_col] % 60).astype(int)
    out["time_hm"] = (
        out["hour"].astype(str).str.zfill(2) + ":" +
        out["minute"].astype(str).str.zfill(2)
    )

    return out.sort_values(minute_col).reset_index(drop=True)


def top_minutes_mae(
    mae_pct: pd.DataFrame,
    col: str,
    top_n: int = 30
) -> pd.DataFrame:
    """
    Devuelve el top horarios según un percentil del MAE (p70 o p80).
    """
    cols_show = [
        "hour", "minute", "time_hm",
        "n", col
    ]

    top_df = (
        mae_pct
        .sort_values(col, ascending=False)
        [cols_show]
        .head(top_n)
        .copy()
    )

    return top_df

def optimal_window_from_mae_rankings(*tops: pd.DataFrame):
    """
    Calcula la intersección horaria robusta a partir de rankings de MAE.
    """
    starts = []
    ends = []

    for df in tops:
        mod = df["hour"] * 60 + df["minute"]
        starts.append(mod.min())
        ends.append(mod.max())

    start_opt = max(starts)
    end_opt = min(ends)

    if start_opt > end_opt:
        raise ValueError("No hay intersección horaria entre rankings de MAE.")

    return {
        "start_minute_of_day": start_opt,
        "end_minute_of_day": end_opt,
        "start_hhmm": f"{start_opt // 60:02d}:{start_opt % 60:02d}",
        "end_hhmm": f"{end_opt // 60:02d}:{end_opt % 60:02d}",
    }

def filter_window(df, start_hhmm, end_hhmm):
    start = int(start_hhmm.split(":")[0]) * 60 + int(start_hhmm.split(":")[1])
    end = int(end_hhmm.split(":")[0]) * 60 + int(end_hhmm.split(":")[1])
    return df[(df["minute_of_day"] >= start) & (df["minute_of_day"] <= end)].copy()

def stoploss_report_df(
    stop_loss_60_p70, stop_loss_60_p80,
    stop_loss_90_p70, stop_loss_90_p80,
    window_60=("09:13","09:41"),
    window_90=("09:09","09:36"),
    decimals=2
):
    report = pd.DataFrame([
        {"horizon_min": 60, "optimal_window": f"{window_60[0]}–{window_60[1]}", "stop_p70_pts": stop_loss_60_p70, "stop_p80_pts": stop_loss_60_p80},
        {"horizon_min": 90, "optimal_window": f"{window_90[0]}–{window_90[1]}", "stop_p70_pts": stop_loss_90_p70, "stop_p80_pts": stop_loss_90_p80},
    ])

    # Redondeo para presentación
    report["stop_p70_pts"] = report["stop_p70_pts"].round(decimals)
    report["stop_p80_pts"] = report["stop_p80_pts"].round(decimals)

    return report





def build_stage03a_summary_table_with_rr_and_stop(
    # deltas H=60
    delta_base_med_60, delta_target_p70_60, delta_tail_p90_60,
    # deltas H=90
    delta_base_med_90, delta_target_p70_90, delta_tail_p90_90,
    # ventanas óptimas (HH:MM)
    window_60=("09:13","09:41"),
    window_90=("09:09","09:36"),
    # stop loss (MAE hasta TP=delta_op)
    stop_loss_60_p70=None, stop_loss_60_p80=None,
    stop_loss_90_p70=None, stop_loss_90_p80=None,
    # regla de recomendación
    rr_min: float = 2.0,
    decimals: int = 2
) -> pd.DataFrame:
    """
    Tabla resumen con:
      - deltas base/target/tail por horizonte
      - ventana óptima por horizonte
      - stop loss p70/p80 por horizonte
      - R/R en 4 combinaciones
      - stop_recomendado:
          * elige SL_p70 si (delta_target_p70 / SL_p70) >= rr_min
          * si no, elige SL_p80
        y RR_recomendado correspondiente.
    """

    rows = [
        {
            "horizon_min": 60,
            "optimal_window": f"{window_60[0]}–{window_60[1]}",
            "delta_base_med": delta_base_med_60,
            "delta_target_p70": delta_target_p70_60,
            "delta_tail_p90": delta_tail_p90_60,
            "stop_loss_p70": stop_loss_60_p70,
            "stop_loss_p80": stop_loss_60_p80,
        },
        {
            "horizon_min": 90,
            "optimal_window": f"{window_90[0]}–{window_90[1]}",
            "delta_base_med": delta_base_med_90,
            "delta_target_p70": delta_target_p70_90,
            "delta_tail_p90": delta_tail_p90_90,
            "stop_loss_p70": stop_loss_90_p70,
            "stop_loss_p80": stop_loss_90_p80,
        },
    ]

    df = pd.DataFrame(rows)

    # Asegurar floats
    num_cols = ["delta_base_med", "delta_target_p70", "delta_tail_p90", "stop_loss_p70", "stop_loss_p80"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # R/R (Reward / Risk)
    df["RR_target_p70_vs_SL_p70"] = df["delta_target_p70"] / df["stop_loss_p70"]
    df["RR_target_p70_vs_SL_p80"] = df["delta_target_p70"] / df["stop_loss_p80"]
    df["RR_tail_p90_vs_SL_p70"]   = df["delta_tail_p90"]   / df["stop_loss_p70"]
    df["RR_tail_p90_vs_SL_p80"]   = df["delta_tail_p90"]   / df["stop_loss_p80"]

    # Reglas de recomendación:
    # Preferimos el stop más ajustado (p70) SI cumple el umbral rr_min.
    # Si no cumple, usamos p80 (más holgado). Si alguno está NaN, elegimos el disponible.
    def recommend_stop(row):
        rr_p70 = row["RR_target_p70_vs_SL_p70"]
        sl_p70 = row["stop_loss_p70"]
        sl_p80 = row["stop_loss_p80"]

        # Casos con NaN
        if pd.isna(sl_p70) and pd.isna(sl_p80):
            return pd.Series({"stop_recomendado": np.nan, "stop_recomendado_tipo": "NA", "RR_recomendado": np.nan})

        if pd.isna(sl_p70):
            rr = row["RR_target_p70_vs_SL_p80"]
            return pd.Series({"stop_recomendado": sl_p80, "stop_recomendado_tipo": "p80", "RR_recomendado": rr})

        if pd.isna(sl_p80):
            rr = row["RR_target_p70_vs_SL_p70"]
            return pd.Series({"stop_recomendado": sl_p70, "stop_recomendado_tipo": "p70", "RR_recomendado": rr})

        # Ambos disponibles
        if pd.notna(rr_p70) and rr_p70 >= rr_min:
            return pd.Series({"stop_recomendado": sl_p70, "stop_recomendado_tipo": "p70", "RR_recomendado": rr_p70})
        else:
            rr = row["RR_target_p70_vs_SL_p80"]
            return pd.Series({"stop_recomendado": sl_p80, "stop_recomendado_tipo": "p80", "RR_recomendado": rr})

    rec = df.apply(recommend_stop, axis=1)
    df = pd.concat([df, rec], axis=1)

    # Redondeo
    for c in num_cols + [
        "RR_target_p70_vs_SL_p70",
        "RR_target_p70_vs_SL_p80",
        "RR_tail_p90_vs_SL_p70",
        "RR_tail_p90_vs_SL_p80",
        "stop_recomendado",
        "RR_recomendado",
    ]:
        df[c] = df[c].round(decimals)

    # Orden de columnas final
    df = df[
        ["horizon_min", "optimal_window",
         "delta_base_med", "delta_target_p70", "delta_tail_p90",
         "stop_loss_p70", "stop_loss_p80",
         "RR_target_p70_vs_SL_p70", "RR_target_p70_vs_SL_p80",
         "RR_tail_p90_vs_SL_p70",   "RR_tail_p90_vs_SL_p80",
         "stop_recomendado_tipo", "stop_recomendado", "RR_recomendado"]
    ]

    return df



##########################
def main():
    mnq_intraday = load_mnq_parquet()
    mnq_intraday = add_column_date(mnq_intraday)
    HORIZONS = [30,60, 90,120]
    mnq_intraday_with_deltas = add_future_delta_pts_by_day(mnq_intraday, horizons=[60,90])
    mnq_intraday_with_deltas = add_minute_of_day(mnq_intraday_with_deltas)

    # Asegurar orden temporal
    mnq_intraday_with_deltas = mnq_intraday_with_deltas.sort_index()

    # date (si no existe)
    if "date" not in mnq_intraday_with_deltas.columns:
        mnq_intraday_with_deltas["date"] = mnq_intraday_with_deltas.index.date

    by_time_60 = stats_by_minute_of_day(mnq_intraday_with_deltas, h=60)
    by_time_90 = stats_by_minute_of_day(mnq_intraday_with_deltas, h=90)
    
    #Top 30
    top_med_60 = top_minutes(by_time_60, col="median_abs", top_n=30)
    top_p70_60 = top_minutes(by_time_60, col="p70_abs", top_n=30)
    top_p90_60 = top_minutes(by_time_60, col="p90_abs", top_n=30)

    top_med_90 = top_minutes(by_time_90, col="median_abs", top_n=30)
    top_p70_90 = top_minutes(by_time_90, col="p70_abs", top_n=30)
    top_p90_90 = top_minutes(by_time_90, col="p90_abs", top_n=30)


  
    ## DELTAS

    delta_base_med_60 = delta_midpoint(
        top_med_60["median_abs"].min(),
        top_med_60["median_abs"].max()
        )

    delta_target_p70_60 = delta_midpoint(
        top_p70_60["p70_abs"].min(),
        top_p70_60["p70_abs"].max()
        )

    delta_tail_p90_60 = delta_midpoint(
            top_p90_60["p90_abs"].min(),
            top_p90_60["p90_abs"].max()
        )

    delta_base_med_90 = delta_midpoint(
            top_med_90["median_abs"].min(),
            top_med_90["median_abs"].max()
        )

    delta_target_p70_90 = delta_midpoint(
        top_p70_90["p70_abs"].min(),
        top_p70_90["p70_abs"].max()
    )

    delta_tail_p90_90 = delta_midpoint(
        top_p90_90["p90_abs"].min(),
        top_p90_90["p90_abs"].max()
    )

    ## VENTANA ÓPTIMA DE OPERACIÓN
    optimal_window_60 = optimal_window_from_rankings(
    top_med_60,
    top_p70_60,
    top_p90_60
        )

    optimal_window_90 = optimal_window_from_rankings(
        top_med_90,
        top_p70_90,
        top_p90_90
    )

    ## STOP LOSS

    mae_60 = compute_mae_until_tp_fast(mnq_intraday_with_deltas, h=60, tp_pts=delta_target_p70_60)
    mae_90 = compute_mae_until_tp_fast(mnq_intraday_with_deltas, h=90, tp_pts=delta_target_p70_90)

    mnq_mae = mnq_intraday_with_deltas.join(mae_60).join(mae_90)
    mnq_mae[["mae_tp_60","tp_hit_60","tau_60","mae_tp_90","tp_hit_90","tau_90"]].head()

    # H = 60
    mae_pct_60 = mae_percentiles_by_time(
        mnq_mae,
        mae_col="mae_tp_60",
        percentiles=(0.7, 0.8)
    )

    # H = 90
    mae_pct_90 = mae_percentiles_by_time(
        mnq_mae,
        mae_col="mae_tp_90",
        percentiles=(0.7, 0.8)
    )

    # H = 60
    top_mae_p70_60 = top_minutes_mae(mae_pct_60, col="p70", top_n=30)
    top_mae_p80_60 = top_minutes_mae(mae_pct_60, col="p80", top_n=30)

    # H = 90
    top_mae_p70_90 = top_minutes_mae(mae_pct_90, col="p70", top_n=30)
    top_mae_p80_90 = top_minutes_mae(mae_pct_90, col="p80", top_n=30)

    window_mae_60 = optimal_window_from_mae_rankings(
        top_mae_p70_60,
        top_mae_p80_60
    )

    window_mae_90 = optimal_window_from_mae_rankings(
        top_mae_p70_90,
        top_mae_p80_90
    )

    # Ventanas finales ya definidas
    optimal_sl_window_60 = (optimal_window_60["start_hhmm"], optimal_window_60["end_hhmm"])
    optimal_sl_window_90 = (optimal_window_90["start_hhmm"], optimal_window_90["end_hhmm"])

    mae_pct_60_win = filter_window(mae_pct_60, *optimal_sl_window_60)
    mae_pct_90_win = filter_window(mae_pct_90, *optimal_sl_window_90)

    stop_loss_60_p70 = mae_pct_60_win["p70"].median()
    stop_loss_60_p80 = mae_pct_60_win["p80"].median()

    stop_loss_90_p70 = mae_pct_90_win["p70"].median()
    stop_loss_90_p80 = mae_pct_90_win["p80"].median()

    stop_loss_60_p70, stop_loss_60_p80, stop_loss_90_p70, stop_loss_90_p80

    # Uso
    report_sl = stoploss_report_df(
        stop_loss_60_p70, stop_loss_60_p80,
        stop_loss_90_p70, stop_loss_90_p80
    )
    report_sl


    summary_tbl = build_stage03a_summary_table_with_rr_and_stop(
    delta_base_med_60, delta_target_p70_60, delta_tail_p90_60,
    delta_base_med_90, delta_target_p70_90, delta_tail_p90_90,
    window_60=optimal_window_60,
    window_90=optimal_window_90,
    stop_loss_60_p70=stop_loss_60_p70,
    stop_loss_60_p80=stop_loss_60_p80,
    stop_loss_90_p70=stop_loss_90_p70,
    stop_loss_90_p80=stop_loss_90_p80,
    decimals=2
    )