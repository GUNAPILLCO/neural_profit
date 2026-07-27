# S06 — Predictive Signal Analysis

## 1. Identificación

- **Notebook:** `S06_predictive_signal_analysis.ipynb`
- **Etapa:** Stage 06
- **Función:** validar los datasets predictivos de Stage 05, medir señal fuera de muestra y seleccionar baselines tabulares.
- **Estado:** ejecutada.
- **Alcance:** modelos tabulares por fila; no construye secuencias temporales ni redes neuronales.
- **Decisión vigente posterior:** Stage 07 prioriza el target `opc_p50_h60_tp15_sl10` y arquitecturas secuenciales.

## 2. Posición dentro del pipeline

```text
S00 → S01 → S02 → S03 → S04 → S05 → S06 → Stage 07
```

Entrada:

```text
data/05_mnq_features/predictive_datasets/
```

Inventario recibido:

```text
BAR:
3 targets × 2 feature sets × 2 scopes = 12 datasets

OPC:
3 targets × 3 feature sets × 2 scopes = 18 datasets

Total = 30 datasets
```

Targets principales definidos en S06:

```text
BAR: bar_p50_h60_tp15_sl10
OPC: opc_p50_h60_tp15_sl10
```

Scopes:

```text
all_regimes
regime_3
```

## 3. Auditoría de los datasets

Se validaron los 30 datasets en:

- estructura de columnas;
- índices temporales;
- orden cronológico;
- duplicados;
- NaN e infinitos;
- clases del target;
- columnas prohibidas;
- consistencia entre versiones `full` y `reduced`;
- correspondencia entre `all_regimes` y `regime_3`;
- disponibilidad temporal para los folds.

Resultado:

```text
Datasets válidos: 30
Warnings: 0
Datasets inválidos: 0
```

No se detectaron:

```text
timestamps duplicados
filas duplicadas
NaN
infinitos
columnas futuras explícitas
solapamiento temporal entre train y validation
```

La auditoría confirma la consistencia interna de los archivos recibidos, pero no corrige problemas metodológicos heredados de etapas anteriores.

## 4. Plan experimental

La notebook organiza 30 experimentos en este orden:

```text
1. Targets principales, all_regimes
2. Targets principales, regime_3
3. Targets alternativos, all_regimes
4. Targets alternativos, regime_3
```

Configuraciones principales iniciales:

```text
BAR_full
BAR_reduced
OPC_full
OPC_reduced_level
OPC_reduced_no_level
```

La comparación busca responder:

```text
qué target contiene mayor señal predictiva
qué feature set generaliza mejor
si regime_3 mejora respecto de all_regimes
si los sets reducidos preservan la señal
qué baseline debe avanzar
```

## 5. Validación walk-forward

Folds utilizados:

```text
WF_01
Train: 2020–2021
Validation: 2022

WF_02
Train: 2020–2022
Validation: 2023

WF_03
Train: 2020–2023
Validation: 2024
```

Comprobaciones:

```text
sin solapamiento temporal
orden cronológico correcto
todas las clases esperadas presentes
targets construidos sin cruzar días
```

El año 2025 se utiliza posteriormente como evaluación OOS adicional.

El año 2026 no se utiliza en la notebook.

## 6. Análisis de Mutual Information por fold

Configuración:

```text
muestra máxima por fold: 250.000 filas
random_state: 42
minute_of_day: discreta
regime_id: discreta
```

### 6.1. Señal BAR

La señal se concentra en variables de rango y volatilidad:

```text
rolling_range_pct_60m
rolling_range_pct_30m
rolling_range_pts
vol_ret_1m_30m
bar_range_mean
bar_range_std
```

BAR mantiene una señal relativamente estable entre folds.

### 6.2. Señal OPC

En `OPC_full`, Mutual Information queda dominada por:

```text
rolling_high_60m
rolling_low_60m
rolling_high_90m
rolling_low_90m
```

Estas variables pueden actuar como proxies de:

```text
nivel nominal del índice
año
tendencia secular
régimen histórico
```

Por esta razón, `OPC_reduced_no_level` resulta metodológicamente importante: conserva variables de rango y volatilidad sin depender directamente de niveles absolutos.

### 6.3. Variables de contexto

```text
minute_of_day: señal pequeña pero estable
regime_id: señal muy baja
```

La señal cambia según el contexto intradiario:

- Pre-market muestra mayor relación con rangos porcentuales y niveles.
- Regular presenta mayor relación con rango local y volatilidad de retornos.

## 7. Redundancia

### BAR

`BAR_full` presenta una estructura fuertemente redundante:

```text
22 de 23 variables conectadas en un mismo grupo de correlación
```

### OPC

Los máximos y mínimos rolling de ventanas próximas presentan correlaciones cercanas a uno.

Conclusión:

```text
los sets full contienen información repetida
los sets compactos pueden reducir complejidad sin perder demasiada señal
```

## 8. Feature sets utilizados en S06

En S06, “original” se refiere a la selección derivada de los datasets reducidos de S05 más variables de contexto. No significa utilizar las 116 features originales.

### 8.1. BAR all_regimes compact — 6 features

```text
minute_of_day
regime_id
rolling_range_pct_60m
rolling_range_pts_30m
vol_ret_1m_30m
bar_range_std_30m
```

### 8.2. BAR regime_3 compact — 7 features

```text
minute_of_day
rolling_range_pct_60m
rolling_range_pct_30m
bar_range_mean_30m
bar_range_std_30m
vol_ret_1m_30m
rolling_range_pts_15m
```

### 8.3. OPC all_regimes compact — 4 features

```text
minute_of_day
regime_id
rolling_range_pct_90m
rolling_range_pts_60m
```

### 8.4. OPC regime_3 compact — 5 features

```text
minute_of_day
rolling_range_pct_90m
rolling_range_pts_60m
bar_range_mean_30m
vol_ret_1m_90m
```

S06 incorpora explícitamente `minute_of_day` y, para modelos globales, `regime_id`.

## 9. Preprocesamiento

Variables numéricas:

```text
imputación por mediana
escalado para regresión logística
```

`regime_id`:

```text
codificación one-hot
```

Todas las transformaciones se ajustan dentro del conjunto de entrenamiento de cada fold.

## 10. Modelos baseline

Se evaluaron:

```text
DummyClassifier
LogisticRegression
DecisionTreeClassifier
HistGradientBoostingClassifier
```

### Dummy

Estrategias:

```text
most_frequent
stratified
```

### Logistic Regression

```text
C = 1,0
solver = lbfgs
max_iter = 1000
class_weight = None o balanced
random_state = 42
```

### Decision Tree

```text
max_depth = 6
min_samples_split = 0,01
min_samples_leaf = 0,005
class_weight = None o balanced
```

### HistGradientBoosting

```text
learning_rate = 0,08
max_iter = 150
max_leaf_nodes = 15
min_samples_leaf = 100
l2_regularization = 1
early_stopping = False
```

Durante una ejecución de HGB apareció un `UnicodeDecodeError` en el lector de consola. El entrenamiento finalizó y no se identificó invalidación de resultados.

## 11. Resultados walk-forward de Logistic Regression balanceada

Promedios principales:

| Configuración | Balanced Accuracy | Macro-F1 |
|---|---:|---:|
| BAR all_regimes original | 0,480114 | 0,363292 |
| BAR regime_3 original | 0,505348 | 0,393124 |
| BAR regime_3 compact | 0,503864 | 0,390085 |
| OPC all_regimes original | 0,300779 | 0,229979 |
| OPC regime_3 original | 0,321444 | 0,248480 |
| OPC regime_3 compact | 0,316945 | 0,247741 |

Conclusiones:

```text
BAR es estadísticamente más predecible que OPC.
regime_3 mejora varias métricas en walk-forward.
los sets compactos conservan casi toda la señal.
```

## 12. Selección del tipo de modelo

### Logistic Regression

Seleccionada como baseline principal por:

```text
mayor estabilidad
menor diferencia train-validation
mejor interpretabilidad
menor complejidad
```

### Decision Tree

Conservado como referencia secundaria. No mejora de forma consistente a la regresión logística.

### HistGradientBoosting

Rechazado en esta etapa por:

```text
gaps elevados entre train y validation
sobreajuste
ganancias fuera de muestra pequeñas o inestables
```

### Dummy

Conservado únicamente como referencia mínima.

## 13. Candidatos previos a OOS 2025

### BAR

```text
BAR_regime_3_original
BAR_regime_3_compact
BAR_all_regimes_original
```

### OPC

```text
OPC_regime_3_original
OPC_regime_3_compact
OPC_all_regimes_original
```

Todos utilizan Logistic Regression con `class_weight="balanced"`.

## 14. Evaluación OOS 2025

Entrenamiento:

```text
2020–2024
```

Evaluación:

```text
2025
```

Cantidad de observaciones:

```text
BAR regime_3:
Train 160.539
OOS 39.582

BAR all_regimes:
Train 345.134
OOS 87.051

OPC regime_3:
Train 321.135
OOS 61.517

OPC all_regimes:
Train 641.085
OOS 122.807
```

### 14.1. Métricas globales BAR

| Configuración | Accuracy | Balanced Accuracy | Macro-F1 | Log Loss |
|---|---:|---:|---:|---:|
| regime_3 original | 0,383811 | 0,535645 | 0,387168 | 1,236734 |
| regime_3 compact | 0,383533 | 0,534242 | 0,386658 | 1,234695 |
| all_regimes original | 0,371782 | 0,515178 | 0,370368 | 1,190117 |

### 14.2. Métricas globales OPC

| Configuración | Accuracy | Balanced Accuracy | Macro-F1 | Log Loss |
|---|---:|---:|---:|---:|
| regime_3 original | 0,420046 | 0,353297 | 0,263872 | 1,642015 |
| regime_3 compact | 0,435311 | 0,351463 | 0,268011 | 1,606476 |
| all_regimes original | 0,381745 | 0,316454 | 0,242966 | 1,667153 |

Todas las configuraciones superan al Dummy estratificado en Balanced Accuracy y Macro-F1.

Balanced Accuracy de azar:

```text
BAR: 1/3 ≈ 0,333
OPC: 1/5 = 0,20
```

Sin embargo, ningún modelo supera al Dummy basado en priors en Log Loss.

## 15. Comparación justa de scope

Se comparan modelos globales y específicos sobre exactamente las mismas filas de `regime_3`.

### BAR

```text
Global evaluado en regime_3:
Balanced Accuracy = 0,539379
Macro-F1 = 0,390270

Regime_3 original:
Balanced Accuracy = 0,535645
Macro-F1 = 0,387168
```

El modelo global resulta ligeramente superior, probablemente por disponer de más datos de entrenamiento.

### OPC

```text
Global evaluado en regime_3:
Balanced Accuracy = 0,348901
Macro-F1 = 0,257730

Regime_3 original:
Balanced Accuracy = 0,353297
Macro-F1 = 0,263872

Regime_3 compact:
Balanced Accuracy = 0,351463
Macro-F1 = 0,268011
```

`OPC_regime_3_compact` ofrece el mejor equilibrio entre desempeño y simplicidad.

## 16. Original frente a compacto

### BAR

```text
9 → 7 features
Δ Balanced Accuracy = -0,001403
Δ Macro-F1 = -0,000510
Acuerdo de predicciones = 94,44 %
```

La versión compacta preserva prácticamente todo el rendimiento.

### OPC

```text
8 → 5 features
Δ Balanced Accuracy = -0,001833
Δ Macro-F1 = +0,004139
Mejora de Log Loss ≈ 0,03554
Acuerdo de predicciones = 86,14 %
```

La versión compacta es preferible como baseline de régimen.

## 17. Resultados por clase

### 17.1. Mapeo BAR

```text
-1 = SL_FIRST
 0 = NO_EVENT
 1 = TP_FIRST
```

Comportamiento:

```text
SL_FIRST real ≈ 9,5 %
SL_FIRST predicho ≈ 47 %
Recall SL_FIRST ≈ 85 %
Precision SL_FIRST ≈ 17 %
TP_FIRST real ≈ 58,5 %
TP_FIRST predicho ≈ 23 %
Recall TP_FIRST ≈ 23 %
```

`class_weight="balanced"` genera una fuerte sobrepredicción de `SL_FIRST`.

### 17.2. Mapeo OPC

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

Comportamiento:

```text
NO_TRADE es la clase mejor reconocida.
F1 NO_TRADE ≈ 0,70–0,72.
Recall LONG_TP ≈ 9 %.
Recall SHORT_TP ≈ 21–23 %.
LONG_SL y SHORT_SL son sobrepredichas.
TP suele confundirse con SL.
```

El modelo distingue mejor actividad frente a `NO_TRADE` que TP frente a SL.

## 18. Estabilidad temporal en 2025

Peores meses:

```text
BAR: abril
OPC: marzo
```

Balanced Accuracy aproximada:

```text
BAR abril:
global 0,374393
regime_3 original 0,354466
regime_3 compact 0,353739

OPC marzo:
global 0,215481
regime_3 original 0,209690
regime_3 compact 0,192036
```

El desempeño se recupera después de mayo. No se observa un deterioro monótono, sino cambios fuertes de distribución.

Conclusión:

```text
los promedios anuales ocultan fallos mensuales severos
```

## 19. Probabilidades y confianza

Todos los modelos están:

```text
UNCALIBRATED
```

### BAR

```text
correlación confianza-acierto negativa
aproximadamente -0,11 a -0,17
```

Las predicciones de mayor confianza no son más fiables. La confianza no debe usarse como filtro operativo.

### OPC

```text
correlación confianza-acierto positiva
aproximadamente 0,34–0,43
```

No obstante:

- la alta confianza parece concentrarse en casos fáciles, especialmente `NO_TRADE`;
- con confianza ≥ 0,70 la cobertura es cercana a 1,3 %;
- los modelos no superan al Dummy en Log Loss;
- la probabilidad no puede interpretarse como probabilidad real de éxito.

Antes de aplicar thresholds de confianza se requiere calibración fuera de muestra.

## 20. Selección histórica final de S06

Candidatos marcados para avanzar:

### BAR

```text
BAR_all_regimes_original
10 features
Balanced Accuracy 2025: 0,515178
Macro-F1 2025: 0,370368
```

Features:

```text
bar_range_mean_10m
bar_range_mean_30m
bar_range_std_30m
minute_of_day
regime_id
rolling_range_pct_30m
rolling_range_pct_60m
rolling_range_pts_15m
rolling_range_pts_30m
vol_ret_1m_30m
```

### OPC

```text
OPC_regime_3_compact
5 features
Balanced Accuracy 2025: 0,351463
Macro-F1 2025: 0,268011
```

Features:

```text
minute_of_day
rolling_range_pct_90m
rolling_range_pts_60m
bar_range_mean_30m
vol_ret_1m_90m
```

Candidatos secundarios:

```text
BAR_regime_3_compact
OPC_all_regimes_original
```

## 21. Artefactos generados

Ruta principal:

```text
data/06_predictive_analysis/07_oos_2025/
```

Subcarpetas:

```text
metrics
class_analysis
confusion
temporal_stability
probability
predictions
final_selection
frozen_models
metadata
```

Resultado:

```text
103 archivos en el manifest
6 pipelines logísticos congelados
predicciones OOS 2025
2 candidatos marcados para avanzar
```

Estos artefactos son locales y no deben cargarse completos en Claude.

## 22. Problemas metodológicos heredados

### 22.1. Regímenes

S06 utiliza los regímenes heredados de S01:

```text
Regular: 10:30–15:29
Closing: 15:30–15:59
16:00 asignada a Overnight
```

La convención aprobada posteriormente es:

```text
Regular: 10:30–14:59
Closing: 15:00–16:00
```

Todos los análisis `regime_3` quedan condicionados por esta diferencia.

### 22.2. Selección de features antes del walk-forward

S05 seleccionó targets y features utilizando todo 2020–2024.

Los folds de S06 validan en 2022, 2023 y 2024. Por ello, esos años influyeron indirectamente en la selección previa.

Las métricas deben considerarse exploratorias u optimistas hasta repetir:

```text
selección de features dentro del train de cada fold
```

### 22.3. Uso de 2025

2025 fue utilizado extensamente para:

- comparar candidatos;
- seleccionar configuraciones;
- analizar clases;
- evaluar estabilidad;
- estudiar confianza.

Por lo tanto, 2025 ya no constituye un holdout completamente ciego.

2026 permanece sin utilizar en S06, aunque solo cubre hasta el 17 de abril.

### 22.4. BAR y utilidad operativa

BAR es más predecible, pero:

- no contiene dirección;
- usa una dirección retrospectiva derivada de DIR;
- no representa una señal desplegable de forma independiente;
- su tasa de TP no equivale a rentabilidad.

## 23. Relación con Stage 07

Conclusión metodológica posterior:

```text
BAR = benchmark estadístico
OPC = target operativo prioritario
```

Diseño vigente de Stage 07:

```text
Target: opc_p50_h60_tp15_sl10
Problema: clasificación multiclase de 5 clases
Scope primario inicial: all_regimes
Feature sets:
- OPC_full
- OPC_reduced_level
- OPC_reduced_no_level

Lookbacks:
- 30
- 60
- 90
Primario: 60

Modelos:
- MLP
- CNN1D
- LSTM
- GRU
- TCN

Métrica principal:
- macro_f1
```

La Logistic Regression de S06 debe conservarse como baseline tabular. No sustituye a los modelos secuenciales.

## 24. Estado y acciones pendientes

**Aprobado:**

- auditoría de los 30 datasets;
- folds walk-forward;
- comparación original/compact;
- Logistic Regression como baseline estable;
- evidencia de mayor predictibilidad de BAR;
- evidencia de dificultad operativa de OPC;
- análisis por clase y temporal;
- constatación de que las probabilidades no están calibradas.

**Pendiente:**

1. definir si se corrigen los regímenes antes de entrenar modelos finales;
2. decidir si la selección de features se repetirá dentro de cada fold;
3. no usar 2025 como holdout ciego;
4. reservar 2026 para una evaluación final después de congelar decisiones;
5. mantener `opc_p50_h60_tp15_sl10` como target principal;
6. comparar los tres feature sets OPC en `all_regimes`;
7. conservar `OPC_regime_3_compact` como baseline histórico secundario;
8. evitar usar confianza como regla operativa sin calibración;
9. no mezclar artefactos históricos con datasets upstream corregidos sin regenerar S03–S06.
