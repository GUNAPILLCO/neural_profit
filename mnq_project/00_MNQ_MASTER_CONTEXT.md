# 00 — MNQ MASTER CONTEXT

## 1. Propósito de este documento

Este archivo resume el contexto técnico, metodológico y operativo del proyecto de Machine Learning aplicado al futuro **Micro E-mini Nasdaq-100 (MNQ)**.

Su objetivo es permitir que Claude comprenda rápidamente:

- qué problema se intenta resolver;
- qué datos existen;
- qué etapas ya fueron desarrolladas;
- qué resultados se obtuvieron;
- qué decisiones siguen vigentes;
- qué aspectos deben revisarse antes de continuar;
- cómo utilizar las notas de estudio de los libros incorporados al proyecto.

Este documento no reemplaza los archivos detallados de cada stage. Debe utilizarse como punto de entrada y como mapa general del proyecto.

---

## 2. Objetivo general del proyecto

El proyecto busca desarrollar y evaluar modelos de Machine Learning capaces de extraer señal predictiva de datos intradía del futuro MNQ.

El objetivo no es únicamente obtener buenas métricas estadísticas, sino construir un pipeline reproducible que permita estudiar la relación entre:

```text
datos de mercado
→ estructura temporal
→ targets
→ features
→ modelos
→ probabilidades o señales
→ reglas operativas
→ backtesting
→ costes
→ riesgo
```

La prioridad metodológica es evitar que el rendimiento aparente provenga de:

- leakage;
- selección retrospectiva;
- reutilización excesiva de validaciones;
- errores de timestamp;
- supuestos irreales de ejecución;
- sobreajuste de features, targets o hiperparámetros.

El proyecto puede reconstruirse desde cero si ello mejora la solidez metodológica. El trabajo previo debe conservarse como investigación exploratoria documentada y como fuente de aprendizajes.

---

## 3. Instrumento y alcance de los datos

### Instrumento

```text
Micro E-mini Nasdaq-100 Futures
Símbolo: MNQ
Mercado: CME Globex
```

### Fuente declarada

```text
NinjaTrader
Archivos históricos trimestrales en formato TXT
Separador: ;
Frecuencia: 1 minuto
```

### Cobertura cruda consolidada

```text
Contratos: MNQH20 hasta MNQM26
Archivos fuente: 26
Filas consolidadas: 2.172.640
Columnas: open, high, low, close, volume, contract
Inicio: 2019-12-23 03:01
Fin: 2026-04-17 20:18
```

El índice crudo es `tz-naive`, pero se interpreta como UTC. Esta interpretación debe confirmarse documentalmente con la configuración de exportación o con la fuente original.

### Dataset intradía utilizado por el pipeline histórico

```text
Periodo: 2020-01-02 04:30 → 2026-04-17 16:00
Zona horaria: America/New_York
Frecuencia: 1 minuto
Filas: 1.024.062
Columnas: 9
Jornadas: 1.482
Barras por jornada: 691
```

Columnas:

```text
date
minute_of_day
regime_id
open
high
low
close
volume
contract
```

Horario operativo histórico:

```text
04:30–16:00 America/New_York
```

Las secuencias, features rolling y targets futuros se diseñaron para no cruzar jornadas.

---

## 4. Convención intradía objetivo

La convención actualmente aprobada para segmentar la jornada es:

| regime_id | Régimen | Horario New York |
|---:|---|---|
| 0 | Overnight | 04:30–08:29 |
| 1 | Pre-market | 08:30–09:29 |
| 2 | Opening | 09:30–10:29 |
| 3 | Regular | 10:30–14:59 |
| 4 | Closing | 15:00–16:00 |

Esta convención representa una hipótesis operativa y no debe interpretarse automáticamente como una segmentación económica demostrada.

El pipeline histórico utilizó una clasificación diferente en varios artefactos. Por lo tanto, cualquier reconstrucción debe decidir explícitamente entre:

```text
A. mantener los artefactos históricos solo para comparación;
B. corregir los regímenes y regenerar las etapas dependientes.
```

Para una nueva implementación, la opción preferida es reconstruir con la convención aprobada y documentar el cambio.

---

## 5. Estructura histórica del pipeline

```text
S00 → S01 → S02 → S03 → S04 → S05 → S06 → S07_00
```

### S00 — Raw Data Preparation

Función:

- consolidar los 26 archivos trimestrales;
- extraer el contrato desde el nombre del archivo;
- concatenar y ordenar los datos;
- verificar timestamps duplicados y solapamientos.

Resultado principal:

```text
data/01_raw/mnq_raw.parquet
```

Estado:

- consolidación aprobada;
- documentación y algunas validaciones deben corregirse;
- existe un gap previo a `MNQM25` que debe auditarse.

### S01 — Intraday Data Preparation

Función:

- interpretar timestamps como UTC;
- convertir a `America/New_York`;
- filtrar 04:30–16:00;
- exigir continuidad de un minuto;
- generar `date`, `minute_of_day` y `regime_id`.

Resultado principal:

```text
data/02_mnq_intraday/mnq_intraday.parquet
```

Estado:

- produjo el dataset base usado por todo el pipeline;
- debe revisarse el calendario de mercado;
- deben auditarse las jornadas eliminadas;
- debe corregirse la clasificación de regímenes;
- debe evitarse la reutilización silenciosa de un Parquet antiguo.

### S02 — Intraday Data Analysis

Función:

- analizar cobertura y calidad;
- estudiar métricas OHLCV;
- construir ventanas históricas y futuras;
- comparar horizontes 30, 60 y 90 minutos;
- proponer validación walk-forward.

Hallazgo principal:

```text
60 minutos aparece como un compromiso razonable entre:
- magnitud de movimiento;
- disponibilidad de muestras;
- horizonte operativo.
```

Los análisis globales siguen siendo útiles. Los análisis por régimen deben revisarse si se corrige S01.

### S03 — Threshold Calibration

Función:

- calcular excursiones futuras usando cierres;
- estimar thresholds por horizonte, percentil y régimen;
- definir benchmarks globales y thresholds primarios por régimen.

Configuración histórica:

```text
Horizontes: 30, 60, 90
Percentiles: 40, 50, 60
Método principal: regime_pooled_2020_2024
Variable principal: threshold_common_pts
```

La combinación que adquirió mayor prioridad fue:

```text
p50 + h60 + threshold por régimen
```

### S04 — Operational Target Investigation

Construyó tres familias de targets:

```text
DIR
BAR
OPC
```

#### DIR

Clasificación direccional basada en excursiones futuras:

```text
1  = LONG
-1 = SHORT
0  = NO_SIGNAL
```

#### BAR

Evalúa si una barrera de TP o SL se alcanza primero, usando la dirección definida retrospectivamente por DIR.

```text
1  = TP_FIRST
-1 = SL_FIRST
0  = NO_EVENT
```

Configuración seleccionada históricamente:

```text
TP = 1,5 × threshold
SL = 1,0 × threshold
```

BAR es útil como benchmark estadístico, pero no constituye por sí solo una señal operativa ex ante porque la dirección fue seleccionada utilizando el futuro.

#### OPC

Combina dirección y resultado de barreras:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

Target principal vigente heredado del pipeline:

```text
opc_p50_h60_tp15_sl10
```

Advertencias:

- `NO_TRADE` combina fenómenos distintos;
- existe fuerte desbalance de clases;
- la dirección proviene de una selección retrospectiva;
- las barreras se evaluaron con cierres de un minuto;
- el alto porcentaje de TP no debe interpretarse como rentabilidad.

### S05 — Feature Engineering and Predictive Datasets

Función:

- construir features causales;
- medir señal univariada;
- definir feature sets;
- generar datasets supervisados.

Resultado:

```text
Features numéricas creadas: 116
Dataset ampliado: 1.024.062 × 147
Datasets predictivos generados: 30
```

Familias de features:

- estructura de la barra;
- retornos y momentum;
- volatilidad y rango;
- posición dentro del rango;
- volumen;
- contexto temporal.

Feature sets OPC históricos:

```text
OPC_full
OPC_reduced_level
OPC_reduced_no_level
```

El set sin niveles es especialmente importante porque reduce el riesgo de que el modelo aprenda el nivel nominal del índice, el año o el contrato.

Advertencia metodológica:

```text
la selección de features utilizó todo 2020–2024 antes de ejecutar folds que evaluaban 2022, 2023 y 2024.
```

Por lo tanto, la selección no fue completamente independiente de los periodos evaluados.

### S06 — Predictive Signal Analysis

Función:

- auditar los 30 datasets;
- ejecutar baselines tabulares;
- medir señal walk-forward;
- evaluar 2025 como periodo OOS adicional;
- seleccionar candidatos históricos.

Modelos evaluados:

```text
DummyClassifier
LogisticRegression
DecisionTreeClassifier
HistGradientBoostingClassifier
```

Conclusiones principales:

- BAR fue más predecible que OPC;
- Logistic Regression balanceada fue el baseline más estable;
- HistGradientBoosting mostró sobreajuste;
- los feature sets compactos preservaron gran parte de la señal;
- OPC distingue mejor `NO_TRADE` que TP frente a SL;
- las probabilidades no están calibradas;
- los promedios anuales ocultan fallos mensuales importantes.

Baseline histórico OPC destacado:

```text
OPC_regime_3_compact
Balanced Accuracy 2025: 0,351463
Macro-F1 2025: 0,268011
```

Sin embargo, los resultados deben considerarse exploratorios porque:

- los regímenes eran los históricos incorrectos;
- las features fueron seleccionadas antes del walk-forward;
- 2025 se utilizó extensamente y dejó de ser holdout ciego.

### S07_00 — Experimental Design

Función:

- definir el protocolo de Stage 07;
- crear configuración, folds, catálogo de modelos y métricas;
- no construye secuencias ni entrena redes.

Configuración primaria histórica:

```text
Target: opc_p50_h60_tp15_sl10
Dataset: OPC_reduced_no_level + all_regimes
Lookback primario: 60
Lookbacks candidatos: 30, 60, 90
Métrica principal: macro_f1
Semilla: 42
```

Modelos previstos:

```text
Dummy
tabular baseline
MLP
CNN1D
GRU
LSTM
TCN
```

Stage 07 todavía no debe considerarse definitivo porque antes deben corregirse o confirmar:

- mapping real de clases;
- columnas físicas del dataset;
- lista exacta de features;
- estado real de 2025 y 2026;
- definición de regímenes;
- uso de class weights;
- implementación de Macro-F1 para early stopping;
- formato y memoria de las secuencias.

---

## 6. Diseño temporal histórico

Folds walk-forward utilizados:

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

Diseño de Stage 07:

1. separar train general y evaluación walk-forward;
2. reservar dentro del train una validación interna cronológicamente posterior;
3. ajustar preprocesamiento solo con train interno;
4. usar validación interna para early stopping y tuning;
5. congelar la configuración;
6. reentrenar con todo el train general;
7. evaluar una sola vez en el fold externo.

### Estado de los periodos posteriores

```text
2025:
- ya fue utilizado en S06;
- no es un test final ciego.

2026:
- no fue utilizado por S06;
- solo cubre hasta el 17 de abril;
- puede reservarse como evaluación final parcial después de congelar decisiones.
```

---

## 7. Modelos de interés

El proyecto no debe limitarse a una única familia de modelos.

### Baselines mínimos

```text
predicción constante
frecuencias del train
reglas simples de momentum o reversión
Logistic Regression
árbol individual
```

### Modelos tabulares

```text
Random Forest
HistGradientBoosting
LightGBM
XGBoost
CatBoost
MLP tabular
```

### Modelos secuenciales

```text
CNN1D
LSTM
GRU
TCN
```

Los modelos tabulares y secuenciales deben evaluarse sobre las mismas muestras finales cuando se realice una comparación directa.

---

## 8. Métricas del proyecto

Las métricas deben mantenerse separadas en tres niveles.

### 8.1. Rendimiento predictivo

Clasificación:

```text
macro F1
balanced accuracy
log loss
accuracy
precision, recall y F1 por clase
matriz de confusión
```

Regresión o señales continuas:

```text
MAE
RMSE
correlación fuera de muestra
IC temporal cuando sea metodológicamente apropiado
```

### 8.2. Calibración

```text
Brier Score
curvas de calibración
error de calibración
confianza por clase
cobertura por threshold
```

Las salidas softmax, sigmoide o las probabilidades de árboles no deben asumirse calibradas.

### 8.3. Rendimiento financiero

```text
P&L bruto
P&L neto
comisiones
spread
slippage
turnover
Sharpe
Sortino
máximo drawdown
Calmar
resultados por operación, sesión, día, fold, año y régimen
```

Una mejora de Macro-F1 no implica automáticamente una mejora financiera.

---

## 9. Reglas de causalidad y leakage

Todas las implementaciones deben respetar:

```text
X_t = información disponible hasta t
y_t = resultado futuro posterior a t
```

Reglas obligatorias:

- ninguna feature puede usar `t+1` o posterior;
- los rolling y lags deben respetar sesiones y contratos;
- las transformaciones deben ajustarse solo con train;
- selección de features, PCA, escalado, imputación y calibración deben ejecutarse dentro del train permitido;
- ninguna secuencia puede cruzar días, gaps, folds o contratos prohibidos;
- la evaluación externa no debe utilizarse para tuning o early stopping;
- la señal generada con el cierre de una barra no debe ejecutarse al mismo cierre sin una justificación realista;
- cualquier operación con TP/SL debe considerar la ambigüedad intrabar de OHLCV de un minuto.

---

## 10. Limitaciones de los datos OHLCV de un minuto

Con los datos actuales no se conoce:

- la secuencia exacta de precios dentro del minuto;
- el orden temporal entre high y low;
- bid y ask;
- spread real;
- profundidad del libro;
- trades individuales;
- dirección agresora;
- fills parciales;
- slippage real;
- VWAP intraminuto verdadero.

Por ello:

- no pueden reconstruirse fielmente barras de ticks, imbalance o runs;
- TP y SL dentro de la misma vela pueden ser ambiguos;
- `(O + H + L + C) / 4` no es VWAP;
- indicadores OHLCV no deben interpretarse como order flow real;
- un motor de backtesting por eventos no elimina la falta de granularidad.

---

## 11. Riesgos metodológicos prioritarios

Antes de continuar deben tratarse explícitamente:

1. calendario de mercado incorrecto o no confirmado;
2. regímenes intradía inconsistentes;
3. jornadas eliminadas sin distinguir gaps de cierres especiales;
4. timestamps UTC inferidos, pero no confirmados documentalmente;
5. continuidad y rollover entre contratos;
6. selección de features antes de los folds;
7. reutilización de 2025 como evaluación y selección;
8. mapping de clases inconsistente entre S04 y S07;
9. desbalance extremo de OPC;
10. `NO_TRADE` como clase heterogénea;
11. targets basados en dirección retrospectiva;
12. barreras evaluadas solo con cierres;
13. niveles absolutos que pueden actuar como proxies temporales;
14. probabilidades sin calibrar;
15. ejecución y costes aún no modelados de forma realista.

Los detalles completos deben documentarse en:

```text
02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
```

---

## 12. Conocimiento proveniente de libros

El proyecto incorpora notas de estudio adaptadas de tres fuentes principales:

```text
Machine Learning for Algorithmic Trading — Stefan Jansen
Advances in Financial Machine Learning — Marcos López de Prado
Technical Analysis of the Financial Markets — John J. Murphy
```

Las notas `.md` asociadas no son simples resúmenes. Deben distinguir:

1. conocimiento explícito del autor;
2. implicaciones validadas para MNQ;
3. contenido no transferible directamente;
4. propuestas experimentales;
5. decisiones pendientes;
6. riesgos metodológicos.

### Regla de interpretación

Las propuestas de los libros no constituyen decisiones automáticas del proyecto.

Ejemplos:

- PCA no debe aplicarse solo porque reduzca varianza;
- GARCH no implica predicción direccional;
- CNN1D no está justificada únicamente por existir dependencia local;
- SHAP no demuestra causalidad;
- Kelly no debe utilizarse sin probabilidades y payoffs robustos;
- CUSUM no sustituye automáticamente las barras de un minuto;
- técnicas multiactivo no se trasladan directamente a un único futuro.

---

## 13. Jerarquía documental para Claude

Claude debe interpretar las fuentes en este orden:

```text
1. 01_CURRENT_DECISIONS.md
2. 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
3. 00_MNQ_MASTER_CONTEXT.md
4. 03_REBUILD_PLAN.md
5. contextos S00–S07
6. notas de estudio de los libros
7. libros completos
```

Ante una contradicción:

- prevalece la decisión vigente más reciente;
- luego la documentación específica del problema;
- después el contexto maestro;
- finalmente los antecedentes históricos y las fuentes bibliográficas.

Claude no debe tratar una propuesta experimental como una decisión adoptada.

---

## 14. Estado actual del proyecto

### Trabajo disponible

```text
S00–S07_00 documentados en archivos CONTEXT.md
Notas de estudio ML4AT disponibles en Markdown
Notas de estudio AFML disponibles en Markdown
Notas de Murphy en proceso
Libros completos disponibles como referencia
```

### Implementación histórica

Existe una implementación completa hasta el diseño de Stage 07, pero varios artefactos dependen de decisiones que deben revisarse.

### Estrategia recomendada

```text
conservar todo el conocimiento
→ auditar las decisiones
→ reconstruir el pipeline desde S00
→ congelar datos y targets
→ ejecutar validación limpia
→ comparar baselines y modelos secuenciales
→ realizar backtesting realista después de cerrar el protocolo predictivo
```

No debe continuarse directamente con el entrenamiento final de Stage 07 hasta resolver los problemas upstream prioritarios.

---

## 15. Instrucciones operativas para Claude

Claude debe actuar como asistente técnico de implementación, auditoría y documentación.

Debe:

- leer primero este archivo y las decisiones vigentes;
- mantener separación estricta entre datos, targets, features, modelos y backtesting;
- justificar cambios antes de implementarlos;
- preservar trazabilidad de configuraciones y artefactos;
- evitar modificar múltiples etapas simultáneamente sin aislar efectos;
- registrar resultados negativos;
- usar rutas relativas y configuraciones portables;
- verificar archivos, columnas, tipos y mappings reales antes de asumirlos;
- distinguir resultados históricos de resultados regenerados;
- detener una etapa cuando exista una inconsistencia upstream no resuelta;
- proponer experimentos controlados, no búsquedas abiertas sin presupuesto.

Claude no debe:

- asumir que los artefactos históricos son definitivos;
- reutilizar 2025 como holdout ciego;
- interpretar probabilidades no calibradas como probabilidades reales;
- utilizar columnas metadata como predictores sin autorización explícita;
- seleccionar hiperparámetros con evaluación externa;
- ejecutar al mismo cierre utilizado para construir la señal;
- confundir buenos resultados estadísticos con rentabilidad;
- copiar implementaciones antiguas de los libros sin adaptarlas al entorno actual.

---

## 16. Resultado esperado de la reconstrucción

La reconstrucción debe producir un pipeline en el que cada etapa tenga:

```text
objetivo
entradas
salidas
supuestos
validaciones
configuración versionada
artefactos reproducibles
criterios de aprobación
problemas pendientes
```

El flujo deseado es:

```text
S00 — consolidación cruda
S01 — preparación temporal y calendario
S02 — auditoría y análisis intradía
S03 — thresholds
S04 — targets
S05 — features y datasets
S06 — baselines y señal predictiva
S07 — secuencias y modelos
S08 — comparación y calibración
S09 — reglas operativas y backtesting
S10 — riesgo, costes y evaluación final
```

La numeración posterior puede ajustarse, pero la separación lógica debe mantenerse.

---

## 17. Resumen ejecutivo

El proyecto dispone de una base sólida de datos, análisis exploratorio, targets, features, baselines y diseño experimental. También dispone de documentación suficiente para reconstruir el trabajo sin perder conocimiento.

Sin embargo, la implementación histórica no debe considerarse definitiva debido a problemas acumulados en calendario, regímenes, selección de features, uso de holdouts y semántica del target OPC.

La decisión recomendada es:

```text
reiniciar la implementación desde S00
sin reiniciar el aprendizaje acumulado.
```

El trabajo previo debe utilizarse como:

- mapa de decisiones;
- inventario de artefactos;
- registro de resultados exploratorios;
- catálogo de errores a evitar;
- base para definir un protocolo más riguroso.

