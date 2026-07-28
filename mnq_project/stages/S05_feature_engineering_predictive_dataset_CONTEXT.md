# S05 — Feature Engineering and Predictive Datasets

## 1. Identificación

- **Notebook:** `S05_feature_engineering_predictive_dataset.ipynb`
- **Etapa:** Stage 05
- **Función:** construir features causales, analizar señal univariada y generar datasets supervisados para Stage 06.
- **Estado:** ejecutada y utilizada por Stage 06.
- **Decisión vigente posterior:** Stage 07 prioriza `opc_p50_h60_tp15_sl10`; las decisiones internas de S05 deben interpretarse como antecedentes exploratorios.

## 2. Posición dentro del pipeline

```text
S00 → S01 → S02 → S03 → S04 → S05 → Stage 06
```

Entradas principales:

```text
data/02_mnq_intraday/mnq_intraday.parquet
data/04_mnq_targets/mnq_targets_selected_final.parquet
data/04_mnq_targets/mnq_targets_stage_decision.json
```

También se cargan, para controles iniciales:

```text
mnq_targets_dir.parquet
mnq_targets_bar.parquet
mnq_targets_opc.parquet
```

Alineación comprobada:

```text
1.024.062 timestamps comunes
Índices exactos, únicos y ordenados
Timezone: America/New_York
Periodo: 2020-01-02 04:30 → 2026-04-17 16:00
```

Dataset base Stage 05:

```text
Shape: 1.024.062 × 31
11 columnas base/metadata
20 columnas target
```

## 3. Objetivo y supuesto operativo

Cada fila representa una señal calculada después del cierre de la barra de un minuto en `t`.

```text
X_t = información disponible hasta t
y_t = resultado futuro asociado a t
```

Se consideran disponibles en `t`:

```text
open, high, low, close, volume
minute_of_day, regime_id, contract
```

Reglas:

- ninguna feature puede usar `t+1` o posterior;
- ningún target puede utilizarse como feature;
- lags y rolling se calculan dentro de cada `date`;
- las ventanas históricas no cruzan jornadas;
- `date`, `year` y `dataset_split` son metadata, no predictores;
- `contract` se conserva como metadata y no genera features en esta versión.

## 4. Targets recibidos

S05 recibe cinco familias preseleccionadas de S04:

```text
DIR: 5
BAR: 5
OPC label: 5
OPC numérico: 5
Total: 20
```

La codificación OPC heredada de S04 es:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

S05 construye los datasets OPC con la columna numérica, no con la etiqueta textual.

## 5. Features causales construidas

Se crean **116 features numéricas**. El dataset ampliado queda en:

```text
1.024.062 × 147
```

### 5.1. Barra actual

```text
bar_range_pts
bar_body_pts
bar_body_abs_pts
upper_wick_pts
lower_wick_pts
close_to_high_pts
close_to_low_pts
bar_range_pct
bar_body_pct
bar_body_abs_pct
```

### 5.2. Retornos y momentum

Lags:

```text
1, 3, 5, 10, 15, 30, 60 y 90 minutos
```

Para cada lag:

```text
ret_pts_*m
ret_pct_*m
```

### 5.3. Volatilidad y rango histórico

Ventanas:

```text
5, 10, 15, 30, 60 y 90 minutos
```

Variables:

```text
vol_ret_1m_*m
bar_range_mean_*m
bar_range_std_*m
rolling_high_*m
rolling_low_*m
rolling_range_pts_*m
rolling_range_pct_*m
```

### 5.4. Posición dentro del rango

```text
close_pos_range_*m
close_dist_high_*m
close_dist_low_*m
```

`close_pos_range` y `close_dist_low` son exactamente equivalentes; `close_dist_high` es su complemento.

### 5.5. Volumen

```text
volume_log
volume_mean_*m
volume_std_*m
volume_rel_*m
volume_zscore_*m
```

### 5.6. Tiempo

```text
day_of_week
hour
minute
minute_of_day_sin
minute_of_day_cos
```

Estas variables temporales fueron construidas, pero no quedaron dentro de los feature sets predictivos finales.

## 6. Validación de calidad

La auditoría se ejecuta principalmente sobre:

```text
development_2020_2024
818.835 filas
```

Resultado:

```text
Features numéricas: 116
Infinitos: 0
Constantes: 0
Casi constantes: 0
Std = 0: 0
Features excluidas por calidad: 0
Máximo ratio de NaN: ≈ 13,02 %
```

Los NaN corresponden al inicio de cada jornada por lags y rolling sin cruce de días.

Redundancia:

```text
Umbral |Spearman|: 0,95
Pares altamente correlacionados: 139
Features involucradas: 87
```

La redundancia se concentra en:

- niveles rolling de ventanas cercanas;
- posición/distancia dentro del rango;
- variantes cercanas de rango y volatilidad.

## 7. Corrección interna de identificación de targets

En el punto 6, la regla:

```python
col.startswith("bar_")
```

clasifica erróneamente features como `bar_range_*` dentro de los targets y reporta 38 targets.

Antes del análisis de señal, la notebook corrige el problema identificando los targets desde `df_stage05_base` y usando:

```python
col.startswith("bar_p")
```

Resultado correcto:

```text
5 DIR + 5 BAR + 5 OPC numéricos = 15 targets analizados
```

El error afecta reportes y registros intermedios del punto 6, pero no la construcción de las 116 features ni el análisis posterior corregido.

## 8. Análisis univariado de señal

Se utiliza exclusivamente `development_2020_2024`.

### 8.1. Spearman / IC

Aplicado a DIR y BAR.

Resultados principales:

```text
DIR: IC máximo aproximado 0,01–0,015
BAR: IC máximo aproximado 0,09–0,12
```

Target BAR con mayor señal global:

```text
bar_p50_h30_tp15_sl10
Top feature: bar_range_mean_10m
IC ≈ 0,120
```

Las features BAR dominantes son:

```text
bar_range_mean
bar_range_std
vol_ret_1m
rolling_range_pts
rolling_range_pct
```

### 8.2. Mutual Information

Aplicada a DIR, BAR y OPC:

```text
15 targets
116 features
50.000 observaciones aleatorias por target
1.740 relaciones feature-target
random_state = 42
```

Los rankings están dominados por:

```text
rolling_high_60m / 90m
rolling_low_60m / 90m
```

Esto indica señal, pero también riesgo de que MI capture:

- nivel absoluto del índice;
- año o tendencia secular;
- cambios de régimen;
- no estacionariedad.

### 8.3. Análisis por régimen

El mayor IC aparece en `regime_id = 3`.

Ejemplo:

```text
bar_p50_h30_tp15_sl10
bar_range_mean_30m
IC ≈ 0,212
```

Este resultado motivó la creación de datasets `regime_3`.

### 8.4. Análisis por año

El Top 50 inicial quedó dominado por 2020. Luego se comparó:

```text
development completo
development sin 2020
regime_3 completo
regime_3 sin 2020
```

BAR conserva aproximadamente 89,8 %–97,1 % del promedio Top 10 al excluir 2020.

OPC presenta MI mayor al excluir 2020 y dentro de régimen 3.

Estas comprobaciones son exploratorias. No sustituyen una validación fuera de muestra ni una selección de features dentro de cada fold.

## 9. Decisiones históricas de S05

### DIR

```text
dir_p50_h60
Rol: benchmark
```

### BAR conservados

```text
bar_p50_h30_tp15_sl10
bar_p50_h60_tp15_sl10
bar_p60_h60_tp15_sl10
```

### OPC conservados

```text
opc_p50_h90_tp15_sl10
opc_p40_h60_tp15_sl10
opc_p50_h60_tp15_sl10
```

S05 consideró BAR como target predictivo principal y OPC como target operativo avanzado.

Esta decisión fue reemplazada posteriormente por Stage 06:

```text
Target principal Stage 07:
opc_p50_h60_tp15_sl10
```

## 10. Feature sets finales

### BAR full — 22 features

Incluye múltiples ventanas de:

```text
bar_range_mean
bar_range_std
vol_ret_1m
rolling_range_pts
rolling_range_pct
```

### BAR reduced — 8 features

```text
bar_range_mean_10m
bar_range_mean_30m
bar_range_std_30m
vol_ret_1m_30m
rolling_range_pts_15m
rolling_range_pts_30m
rolling_range_pct_30m
rolling_range_pct_60m
```

### OPC full — 18 features

```text
rolling_high_30m
rolling_low_30m
rolling_high_60m
rolling_low_60m
rolling_high_90m
rolling_low_90m
rolling_range_pct_30m
rolling_range_pct_60m
rolling_range_pct_90m
rolling_range_pts_30m
rolling_range_pts_60m
rolling_range_pts_90m
bar_range_mean_10m
bar_range_mean_15m
bar_range_mean_30m
vol_ret_1m_30m
vol_ret_1m_60m
vol_ret_1m_90m
```

### OPC reduced level — 8 features

```text
rolling_high_90m
rolling_range_pct_60m
rolling_range_pct_90m
rolling_range_pts_60m
rolling_range_pts_90m
bar_range_mean_30m
vol_ret_1m_60m
vol_ret_1m_90m
```

### OPC reduced no level — 7 features

```text
rolling_range_pct_60m
rolling_range_pct_90m
rolling_range_pts_60m
rolling_range_pts_90m
bar_range_mean_30m
vol_ret_1m_60m
vol_ret_1m_90m
```

Advertencia: `full` significa “set completo preseleccionado”, no las 116 features construidas.

## 11. Datasets predictivos generados

Combinaciones:

```text
BAR:
3 targets × 2 feature sets × 2 regímenes = 12

OPC:
3 targets × 3 feature sets × 2 regímenes = 18

Total = 30 datasets
```

Versiones:

```text
all_regimes
regime_3
```

Columnas incluidas:

```text
metadata
features seleccionadas
target numérico
```

Metadata:

```text
date
year
dataset_split
regime_id
minute_of_day
contract
```

Se eliminan filas con NaN en cualquier feature seleccionada o en el target.

### Dataset actualmente prioritario

Para `opc_p50_h60_tp15_sl10`:

```text
all_regimes:
801.762 filas para los tres feature sets
development: 641.085
final_test: 160.677

regime_3:
401.622 filas para los tres feature sets
development: 321.135
final_test: 80.487
```

La pérdida adicional respecto de las 935.142 etiquetas válidas de H60 se debe principalmente a que los feature sets OPC incluyen ventanas históricas de 90 minutos.

## 12. Distribución de clases

BAR:

```text
3 clases: -1, 0, 1
Desbalance moderado
```

OPC:

```text
5 clases: 0, 1, 2, 3, 4
Desbalance fuerte o severo
```

En development y `regime_3`, algunos OPC presentan:

```text
clase minoritaria ≈ 0,98 %
clase dominante ≈ 73 %
imbalance ratio ≈ 74
```

Consecuencias:

- accuracy no debe ser métrica principal;
- usar macro-F1, balanced accuracy y métricas por clase;
- evaluar matriz de confusión;
- aplicar class weights u otras técnicas solo dentro de train.

## 13. Artefactos generados

```text
data/05_mnq_features/
├── predictive_datasets/   # 30 Parquet
├── summaries/
│   ├── stage05_predictive_datasets_summary.parquet/csv
│   ├── stage05_class_distribution.parquet/csv
│   └── stage05_class_balance_summary.parquet/csv
└── registries/
    └── stage05_final_registry.json
```

El registro JSON contiene rutas absolutas de la computadora local. Para portabilidad, deberían guardarse rutas relativas al proyecto.

## 14. Problemas y advertencias críticas

### 14.1. Regímenes heredados

Los datasets usan la clasificación antigua de S01:

```text
Regular: 10:30–15:29
Closing: 15:30–15:59
16:00 asignada a Overnight
```

Por eso `regime_3` contiene:

```text
444.600 filas antes de dropna
300 barras por jornada
```

La convención aprobada posteriormente es:

```text
Regular: 10:30–14:59
Closing: 15:00–16:00
```

Todos los datasets `regime_3` y los análisis por régimen quedan afectados.

### 14.2. Selección de features antes del walk-forward

Las features y targets se seleccionan usando todo `development_2020_2024`.

Posteriormente, los folds validan en 2022, 2023 y 2024. Por lo tanto, la selección de variables ya vio indirectamente esos años antes de evaluarlos.

Para métricas walk-forward estrictamente no sesgadas, la selección de features debe:

```text
ejecutarse dentro del train de cada fold
```

o tratar los resultados actuales de Stage 06 como exploratorios.

### 14.3. “Robustez” no equivale a generalización

Excluir 2020 y analizar régimen 3 no constituye una prueba fuera de muestra. MI se calcula sobre muestras aleatorias de 50.000 filas y puede variar por composición de la muestra.

La estabilidad debe confirmarse mediante folds temporales y modelos reales.

### 14.4. Nivel absoluto del precio

`rolling_high` y `rolling_low` dominan MI y están casi perfectamente correlacionados entre ventanas. Pueden identificar épocas históricas en lugar de patrones transferibles.

La comparación:

```text
OPC_full
OPC_reduced_level
OPC_reduced_no_level
```

es obligatoria.

### 14.5. Metadata dentro de los archivos

Los Parquet contienen `year`, `dataset_split`, `contract`, `regime_id` y `minute_of_day`.

No deben seleccionarse automáticamente todas las columnas salvo el target. En particular:

```text
dataset_split y year están prohibidos como features
contract puede codificar indirectamente el periodo histórico
```

### 14.6. Contexto intradiario no incluido en los feature sets

Aunque se construyeron variables temporales y se identificó dependencia por régimen, los feature sets finales no incluyen:

```text
regime_id
minute_of_day_sin
minute_of_day_cos
day_of_week
```

En `all_regimes`, el modelo no conoce explícitamente el régimen salvo que Stage 06/07 lo incorpore de forma controlada.

### 14.7. BAR no es directamente desplegable

Los datasets BAR eliminan filas donde DIR no define una dirección. DIR fue determinada retrospectivamente con información futura en S04.

BAR puede usarse como benchmark predictivo, pero no representa por sí solo una señal operable ex ante.

### 14.8. Holdout final observado

S05 no usa 2025–2026 para calcular IC o MI, pero inspecciona sus distribuciones de clases. Esto reduce parcialmente la pureza conceptual del holdout.

### 14.9. Mapeo de clases

El registro final de S05 no documenta claramente el mapeo OPC. Stage 07 debe leer la metadata de S04 y verificar los valores reales antes de entrenar.

## 15. Relación con el estado actual

Elementos vigentes:

```text
features causales sin cruzar días
feature sets OPC full/reduced_level/reduced_no_level
all_regimes como configuración primaria
regime_3 como experimento secundario
macro-F1 y balanced accuracy por desbalance
opc_p50_h60_tp15_sl10 como target principal
```

Elementos históricos o exploratorios:

```text
BAR como objetivo principal
regime_3 como contexto principal
selección univariada sobre todo 2020–2024
opc_p50_h90 y opc_p40_h60 como candidatos prioritarios
```

Antes del entrenamiento definitivo debe decidirse si se mantiene el pipeline histórico para comparación o si se corrigen regímenes y selección fold-specific, lo que requeriría regenerar artefactos posteriores.
