# S07_00 — Experimental Design

## 1. Identificación

- **Notebook:** `S07_00_experimental_design.ipynb`
- **Etapa:** Stage 07 — diseño experimental
- **Función:** definir el protocolo común del Stage 07 y crear sus archivos centrales de configuración.
- **Composición:** 420 celdas; 415 Markdown y 5 de código.
- **Estado:** ejecutada; la estructura de carpetas y los archivos de configuración fueron creados y verificados.
- **Alcance real:** no carga datasets, no construye secuencias y no entrena modelos.

## 2. Posición dentro del pipeline

```text
S00 → S01 → S02 → S03 → S04 → S05 → S06 → S07_00 → S07_01
```

S07_00 recibe las decisiones metodológicas de los stages anteriores y centraliza las reglas que deberán leer todas las notebooks de modelos.

Próxima notebook prevista:

```text
S07_01_sequence_dataset.ipynb
```

## 3. Objetivo del Stage 07

Evaluar si una representación secuencial explícita mejora los baselines tabulares para predecir el target OPC.

Relación principal:

```text
[X_(t-L+1), ..., X_t] → y_t
```

donde:

- la entrada utiliza solamente información disponible hasta `t`;
- el target se alinea con el último instante de la ventana;
- el futuro se utiliza únicamente para construir la etiqueta;
- las secuencias no pueden cruzar días, gaps ni cambios de contrato.

El Stage 07 evalúa calidad predictiva. La transformación de probabilidades en operaciones y el backtesting pertenecen a etapas posteriores.

## 4. Configuración experimental principal

```text
Target:
opc_p50_h60_tp15_sl10

Problema:
clasificación multiclase de 5 clases

Dataset:
opc_p50_h60_tp15_sl10__OPC_reduced_no_level__all_regimes

Feature set:
OPC_reduced_no_level

Scope:
all_regimes

Lookbacks candidatos:
30, 60 y 90 minutos

Lookback primario:
60 minutos

Métrica principal:
macro_f1

Semilla:
42
```

La notebook selecciona inicialmente una sola configuración controlada:

```text
OPC_reduced_no_level + all_regimes + lookback 60
```

Las features con nivel, otros feature sets, `regime_3` y los lookbacks 30/90 quedan para experimentos posteriores.

## 5. Target OPC

Configuración:

```text
Target direccional base: dir_p50_h60
Target de barreras: bar_p50_h60_tp15_sl10
Threshold: percentil 50 por régimen
Horizonte futuro: 60 minutos
TP: 1,5 × threshold
SL: 1,0 × threshold
```

Clases conceptuales:

```text
LONG_TP
LONG_SL
SHORT_TP
SHORT_SL
NO_TRADE
```

El modelo debe producir cinco probabilidades.

### Codificación real heredada de S04

La codificación guardada realmente es:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

La configuración generada por S07_00 contiene provisionalmente:

```text
0 = LONG_TP
1 = LONG_SL
2 = SHORT_TP
3 = SHORT_SL
4 = NO_TRADE
```

y marca:

```text
class_mapping_verified = false
```

Esto debe corregirse antes de construir secuencias. No basta con conservar un mapping incorrecto acompañado por una bandera de advertencia, porque una notebook posterior podría utilizarlo accidentalmente.

## 6. Dataset y features

Dataset primario:

```text
opc_p50_h60_tp15_sl10__OPC_reduced_no_level__all_regimes
```

Features base esperadas de `OPC_reduced_no_level`:

```text
rolling_range_pct_60m
rolling_range_pct_90m
rolling_range_pts_60m
rolling_range_pts_90m
bar_range_mean_30m
vol_ret_1m_60m
vol_ret_1m_90m
```

Contexto predictivo adicional:

```text
minute_of_day
regime_id
```

Tratamiento propuesto:

```text
minute_of_day → normalización entre 0 y 1
regime_id → one-hot con categorías [0, 1, 2, 3, 4]
```

Contexto analítico propuesto:

```text
trading_date
year
quarter
contract
```

Columnas de control excluidas:

```text
dataset_split
fold_id
validation_year
target_name
dataset_name
```

La lista definitiva de features queda pendiente:

```text
feature_columns = null
feature_columns_verified = false
```

Debe cargarse desde los metadatos reales de Stage 05/06 y contrastarse contra el archivo Parquet.

### Nombres de columnas pendientes de reconciliación

Los datasets finales de Stage 05 fueron documentados con:

```text
date
year
dataset_split
regime_id
minute_of_day
contract
```

S07_00 utiliza conceptualmente:

```text
trading_date
quarter
```

Es posible que `trading_date` y `quarter` no existan físicamente. S07_01 debe resolver los nombres reales antes de continuar, sin crear silenciosamente una convención distinta.

## 7. Ventanas temporales

Reglas:

```text
Ventana L:
t-L+1, ..., t

Target:
y_t

Futuro del target:
t+1, ..., t+60
```

Una ventana válida debe:

- pertenecer a un único día;
- tener frecuencia continua de un minuto;
- pertenecer a un único contrato;
- terminar exactamente en el timestamp del target;
- no incluir futuro;
- poder cruzar regímenes intradiarios.

Forma común:

```text
X: (n_samples, lookback, n_features)
y: (n_samples,)
```

Uso por arquitectura:

```text
CNN1D/LSTM/GRU/TCN → secuencia 3D
MLP → ventana aplanada
```

### Historia efectiva

Aunque el lookback primario es 60, varias features resumen 60 o 90 minutos históricos. Por ejemplo, una secuencia de 60 observaciones de `rolling_range_pct_90m` puede depender de información que se remonta aproximadamente hasta `t-149`.

Por tanto:

```text
lookback del modelo = 60
historia efectiva máxima ≈ 149 minutos
```

Esto no constituye leakage, pero debe documentarse para interpretar correctamente el alcance temporal del modelo.

### Volumen de datos

La materialización completa de cientos de miles de ventanas 3D puede requerir varios gigabytes.

S07_01 debería evitar copias repetidas y considerar:

```text
memmap
datasets por chunks
generadores
tf.data / PyTorch Dataset
formatos Zarr o HDF5
```

La representación lógica debe ser común, pero no es obligatorio duplicar físicamente todas las ventanas para cada modelo.

## 8. Walk-forward

Folds:

```text
WF_01:
Train general 2020–2021
Validation walk-forward 2022

WF_02:
Train general 2020–2022
Validation walk-forward 2023

WF_03:
Train general 2020–2023
Validation walk-forward 2024
```

Validación interna:

```text
último trimestre del último año del train general
```

Ejemplo WF_01:

```text
Train interno:
2020–septiembre 2021

Validation interna:
octubre–diciembre 2021

Validation walk-forward:
2022
```

Procedimiento:

1. ajustar preprocesamiento con train interno;
2. entrenar y aplicar early stopping con validation interna;
3. identificar la mejor cantidad de épocas;
4. crear un modelo nuevo;
5. reajustar preprocesamiento con todo el train general;
6. entrenar durante la cantidad de épocas seleccionada;
7. evaluar una sola vez en el año walk-forward.

Las divisiones se realizan mediante jornadas completas. Como los targets tampoco cruzan días, no se considera necesario aplicar un embargo adicional entre días consecutivos.

## 9. Test final

S07_00 declara:

```text
2025–2026 = test final reservado
```

Esto no coincide con el historial real:

- S06 utilizó 2025 para comparar configuraciones;
- 2025 intervino en la selección de candidatos;
- se analizaron métricas, meses, clases y confianza sobre 2025.

Por lo tanto:

```text
2025 ya no es un holdout ciego.
```

La configuración debe corregirse. Las opciones metodológicas son:

```text
A. Tratar 2025 como evaluación OOS de desarrollo y reservar 2026.
B. Rediseñar la selección sin usar 2025, si se desea recuperarlo como holdout.
```

Con el pipeline actual, la opción más coherente es reservar solamente 2026, aclarando que es un periodo parcial hasta el 17 de abril.

## 10. Modelos previstos

Catálogo habilitado:

| ID | Modelo | Entrada | Estructura temporal |
|---|---|---|---|
| `dummy` | Dummy Classifier | distribución/labels | No |
| `tabular` | Baseline tabular | fila tabular | No |
| `mlp` | MLP | ventana aplanada | No explícita |
| `cnn1d` | CNN 1D | secuencia | Sí |
| `gru` | GRU | secuencia | Sí |
| `lstm` | LSTM | secuencia | Sí |
| `tcn` | TCN | secuencia causal | Sí |

Notebooks previstas:

```text
S07_01_sequence_dataset.ipynb
S07_02_dummy_baseline.ipynb
S07_03_tabular_baseline.ipynb
S07_04_mlp.ipynb
S07_05_cnn1d.ipynb
S07_06_gru.ipynb
S07_07_lstm.ipynb
S07_08_tcn.ipynb
S07_09_model_comparison.ipynb
```

El Dummy y el baseline tabular deben evaluarse sobre exactamente los mismos timestamps finales de las secuencias, no sobre una cantidad distinta de filas.

## 11. Protocolo de entrenamiento

Configuración guardada:

```text
max_epochs: 50
patience: 5
batch_size: 512
optimizer: Adam
learning_rate: 0,001
loss: sparse_categorical_crossentropy
early stopping: val_macro_f1
class weights: activados
shuffle de batches de train: activado
shuffle de validation: desactivado
```

Todo preprocesamiento y los pesos de clase se calculan únicamente con el train permitido en cada etapa.

### Inconsistencias documentales

El Markdown menciona alternativamente:

```text
categorical_crossentropy
sparse_categorical_crossentropy
```

Como el target se almacena mediante enteros, la opción coherente es:

```text
sparse_categorical_crossentropy
```

También aparecen dos nombres para la métrica de early stopping:

```text
validation_macro_f1
val_macro_f1
```

Debe definirse un único nombre implementado por un callback o métrica personalizada.

### Pesos de clase

El texto indica que los pesos “podrán” utilizarse, pero la configuración los activa obligatoriamente:

```text
use_class_weights = true
```

S06 mostró que el balanceo podía sobrepredecir las clases perdedoras. Los pesos de clase deben tratarse como una variante experimental controlada, no asumirse automáticamente como la única configuración.

## 12. Métricas

Métrica principal:

```text
macro_f1
```

Métricas generales:

```text
macro_f1
balanced_accuracy
log_loss
accuracy
```

Métricas por clase:

```text
precision
recall
f1_score
support
```

Diagnósticos obligatorios:

```text
matriz de confusión
distribución de predicciones
probabilidades
resultados por fold
resultados por año
resultados por régimen
curvas de entrenamiento
```

Criterios de comparación:

- promedio entre folds;
- desviación estándar;
- peor fold;
- desempeño por clase;
- ausencia de colapso hacia `NO_TRADE`;
- calidad probabilística;
- estabilidad temporal;
- simplicidad en caso de empate.

## 13. Artefactos creados

Ruta local:

```text
data/07_mnq_models/
```

Estructura:

```text
config/
sequences/
models/
predictions/
metrics/
reports/
```

Archivos creados y verificados:

```text
config/stage_07_experimental_config.json
config/stage_07_folds.csv
config/stage_07_models.csv
config/stage_07_metrics.csv
```

Resultado de ejecución:

```text
Folds: 3
Modelos habilitados: 7
Métricas registradas: 8
Métrica principal: macro_f1
```

## 14. Problemas de implementación

### 14.1. Búsqueda de la raíz del proyecto

El código busca una carpeta denominada exactamente:

```text
neural_profit_local
```

mediante un bucle ascendente.

Si la notebook se ejecuta fuera de esa estructura y alcanza la raíz del sistema, `parent == current` y el bucle puede quedar infinito.

Debe reemplazarse por una búsqueda segura con condición de salida o una configuración explícita y portable de `PROJECT_ROOT`.

### 14.2. Configuración validada solo internamente

Las validaciones comprueban:

- unicidad de folds, modelos y métricas;
- coherencia básica de años;
- estructura JSON;
- existencia de cinco clases.

No comprueban todavía:

- dataset real;
- nombres de columnas;
- mapping verdadero;
- features;
- categorías reales;
- cantidad de muestras;
- regímenes;
- disponibilidad de años;
- continuidad temporal.

Por eso, “configuración base válida” significa coherencia interna, no validación contra los datos.

### 14.3. Regímenes heredados

El dataset todavía hereda la clasificación histórica:

```text
Regular: 10:30–15:29
Closing: 15:30–15:59
16:00 asignada a Overnight
```

La convención posteriormente aprobada es:

```text
Regular: 10:30–14:59
Closing: 15:00–16:00
```

Como S07_00 incorpora `regime_id` como predictor y el target depende de thresholds por régimen, esta cuestión debe resolverse antes del entrenamiento definitivo.

### 14.4. Selección de features heredada

Las features de S05 se seleccionaron utilizando todo 2020–2024 antes de ejecutar los folds de S06.

Por ello, los resultados walk-forward de Stage 07 no serán completamente independientes respecto de la selección de variables, salvo que esta se repita dentro de cada fold o se declare explícitamente que el feature set queda congelado como una decisión previa.

### 14.5. Semántica del target

OPC hereda de S04:

- dirección seleccionada retrospectivamente mediante DIR;
- barreras evaluadas con cierres de un minuto;
- `NO_TRADE` heterogéneo;
- fuerte desbalance;
- alta tasa de TP inducida parcialmente por el etiquetado.

El modelo puede aprender la etiqueta definida, pero su Macro-F1 no equivale directamente a rentabilidad o capacidad de ejecución real.

## 15. Decisiones vigentes

```text
Target principal:
opc_p50_h60_tp15_sl10

Experimento inicial:
OPC_reduced_no_level + all_regimes

Lookback primario:
60

Arquitecturas:
Dummy, tabular, MLP, CNN1D, GRU, LSTM y TCN

Métrica principal:
macro_f1

Validación:
WF_01, WF_02 y WF_03

Early stopping:
validation interna temporal

Secuencias:
sin cruzar días, gaps ni contratos
```

## 16. Correcciones obligatorias antes de S07_01

1. Sustituir el mapping provisional por el mapping real de S04.
2. Verificar el archivo exacto del dataset y sus columnas.
3. Cargar la lista real de features desde metadata.
4. Resolver `date` frente a `trading_date` y la existencia de `quarter`.
5. Corregir el estado de 2025–2026 como test final.
6. Decidir si los regímenes históricos se corrigen o se congelan conscientemente.
7. Definir si los pesos de clase son obligatorios o una variante.
8. Unificar el nombre y la implementación de Macro-F1 para early stopping.
9. Unificar `sparse_categorical_crossentropy` en código y documentación.
10. Hacer portable y segura la detección de `PROJECT_ROOT`.
11. Estimar memoria y definir el formato físico de las secuencias.
12. Asegurar que todos los baselines utilicen exactamente las mismas muestras.
13. Documentar que el feature set fue seleccionado antes del walk-forward.
14. Actualizar `validation_status` solamente después de validar contra los datos reales.

Hasta completar estos puntos, S07_00 debe considerarse una configuración base implementada, pero no todavía una configuración experimental definitiva.
