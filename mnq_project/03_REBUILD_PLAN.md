# 03 — REBUILD PLAN

## 1. Propósito del documento

Este archivo define el plan oficial para reconstruir el proyecto MNQ desde una base metodológica limpia, utilizando Claude como asistente de desarrollo sin perder el conocimiento acumulado.

La reconstrucción persigue cuatro objetivos:

1. corregir errores upstream antes de entrenar modelos;
2. conservar los hallazgos útiles del pipeline histórico;
3. impedir que decisiones exploratorias se conviertan en reglas definitivas;
4. producir un flujo reproducible desde datos crudos hasta evaluación financiera.

Este documento debe utilizarse junto con:

```text
00_MNQ_MASTER_CONTEXT.md
01_CURRENT_DECISIONS.md
02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
```

Jerarquía:

```text
01_CURRENT_DECISIONS
→ 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS
→ 03_REBUILD_PLAN
→ 00_MNQ_MASTER_CONTEXT
→ S00–S07_CONTEXT
→ notas de estudio
→ libros completos
```

Cuando una etapa histórica contradiga este plan, Claude debe detenerse y señalar la contradicción.

---

# 2. Principios de reconstrucción

## 2.1. Reiniciar implementación, no conocimiento

El nuevo pipeline no parte de ignorancia.

Debe reutilizar:

- estructura de stages;
- decisiones documentadas;
- funciones que hayan sido verificadas;
- validaciones útiles;
- nombres de artefactos cuando no generen ambigüedad;
- resultados exploratorios como hipótesis;
- notas de estudio de Jansen, López de Prado y Murphy.

No debe reutilizar automáticamente:

- Parquet históricos downstream;
- mappings provisionales;
- regímenes antiguos;
- selecciones de features globales;
- métricas históricas como evidencia definitiva;
- 2025 como test ciego;
- configuraciones de Stage 07 sin auditar.

---

## 2.2. Cambios upstream obligan a regenerar downstream

Regla:

```text
si cambia un dato, definición o regla upstream
→ todos los artefactos dependientes deben regenerarse
```

Ejemplos:

```text
cambio en calendario
→ cambia dataset intradía
→ cambian thresholds
→ cambian targets
→ cambian features
→ cambian folds y resultados
```

No se deben combinar artefactos provenientes de versiones incompatibles.

---

## 2.3. Cada etapa tiene una puerta de aceptación

Claude no debe avanzar automáticamente de un stage al siguiente.

Cada stage debe terminar con:

- validaciones;
- resumen;
- manifest;
- decisiones;
- cuestiones abiertas;
- estado de aprobación.

Estados posibles:

```text
DRAFT
REVIEW_REQUIRED
APPROVED
REJECTED
SUPERSEDED
```

Solo un stage `APPROVED` puede alimentar el siguiente.

---

## 2.4. Separación estricta de funciones

Cada notebook debe cumplir una única responsabilidad principal.

Ejemplo:

```text
S01 prepara datos
S03 calibra thresholds
S04 construye targets
S05 construye features
S06 evalúa baselines
S07 entrena modelos
```

Una notebook de modelos no debe modificar:

- datos;
- targets;
- folds;
- reglas de ejecución;
- mappings;
- definiciones de features.

---

# 3. Resultado final esperado

La reconstrucción debe producir:

```text
datos crudos auditados
→ dataset intradía validado
→ análisis exploratorio reproducible
→ thresholds fold-aware
→ targets auditados
→ features causales
→ datasets predictivos versionados
→ baselines temporales
→ modelos tabulares y secuenciales
→ calibración
→ comparación robusta
→ backtest causal
→ reporte final
```

---

# 4. Estructura de carpetas propuesta

```text
mnq_ml_project/
│
├── context/
│   ├── 00_MNQ_MASTER_CONTEXT.md
│   ├── 01_CURRENT_DECISIONS.md
│   ├── 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
│   └── 03_REBUILD_PLAN.md
│
├── study/
│   ├── ML4AT_*.md
│   ├── AFML_*.md
│   └── MURPHY_*.md
│
├── notebooks/
│   ├── S00_raw_data_preparation.ipynb
│   ├── S01_intraday_data_preparation.ipynb
│   ├── S02_intraday_data_analysis.ipynb
│   ├── S03_threshold_calibration.ipynb
│   ├── S04_target_design.ipynb
│   ├── S05_feature_engineering.ipynb
│   ├── S06_tabular_baselines.ipynb
│   ├── S07_sequence_dataset.ipynb
│   ├── S08_model_training.ipynb
│   ├── S09_model_comparison.ipynb
│   ├── S10_probability_calibration.ipynb
│   └── S11_backtesting.ipynb
│
├── src/
│   ├── data/
│   ├── validation/
│   ├── targets/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── backtesting/
│
├── config/
│   ├── project_config.yaml
│   ├── data_config.yaml
│   ├── regime_config.yaml
│   ├── target_config.yaml
│   ├── feature_config.yaml
│   ├── folds_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── 00_source/
│   ├── 01_raw/
│   ├── 02_intraday/
│   ├── 03_thresholds/
│   ├── 04_targets/
│   ├── 05_features/
│   ├── 06_datasets/
│   ├── 07_sequences/
│   ├── 08_models/
│   ├── 09_predictions/
│   ├── 10_metrics/
│   └── 11_backtests/
│
├── reports/
│   ├── stage_reports/
│   ├── model_reports/
│   └── final_report/
│
└── manifests/
    ├── data_manifest.csv
    ├── experiment_registry.csv
    └── artifact_lineage.csv
```

---

# 5. Fase 0 — Preparación del entorno Claude

## Objetivo

Entregar a Claude el contexto correcto antes de generar código.

## Archivos a cargar

### Obligatorios

```text
00_MNQ_MASTER_CONTEXT.md
01_CURRENT_DECISIONS.md
02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
03_REBUILD_PLAN.md
S00–S07_CONTEXT.md
```

### Conocimiento de estudio

```text
ML4AT_*.md
AFML_*.md
MURPHY_*.md
```

### Fuentes secundarias

```text
Machine Learning for Algorithmic Trading
Advances in Financial Machine Learning
Análisis Técnico de los Mercados Financieros
```

Los libros completos deben utilizarse como referencia secundaria, no como fuente principal de decisiones del proyecto.

## Instrucción inicial recomendada para Claude

```text
Actúa como arquitecto técnico y desarrollador principal del proyecto MNQ.

Antes de proponer código:
1. consulta 01_CURRENT_DECISIONS.md;
2. consulta 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md;
3. sigue 03_REBUILD_PLAN.md;
4. trata S00–S07_CONTEXT como documentación histórica;
5. no asumas que los artefactos históricos son definitivos;
6. señala contradicciones antes de avanzar;
7. no modifiques decisiones upstream desde notebooks downstream;
8. registra toda decisión nueva y su impacto.
```

## Criterio de aceptación

Claude debe poder responder correctamente:

- cuál es la zona horaria;
- cuál es el horario objetivo;
- cuáles son los regímenes vigentes;
- por qué 2025 no es holdout;
- por qué BAR no es una señal operativa;
- cuál es el mapping OPC real;
- por qué Stage 07 histórico no está listo.

---

# 6. Fase 1 — Gobernanza y configuración central

## Objetivo

Crear las reglas centrales antes de procesar datos.

## Artefactos

```text
config/project_config.yaml
config/data_config.yaml
config/regime_config.yaml
config/folds_config.yaml
manifests/artifact_lineage.csv
manifests/experiment_registry.csv
```

## Decisiones que deben declararse

- timezone de origen;
- timezone operativo;
- horario operativo;
- calendario;
- política de días especiales;
- regímenes;
- convención de contrato;
- random seed;
- rutas relativas;
- política de sobrescritura;
- versionado de artefactos;
- estado de 2025 y 2026.

## Validaciones

- todos los archivos de configuración cargan;
- no existen rutas absolutas;
- ningún mapping está duplicado en notebooks;
- los regímenes suman exactamente el horario operativo;
- la barra 16:00 pertenece a Closing;
- los folds están ordenados temporalmente.

## Puerta de aceptación

```text
No avanzar a S00 hasta aprobar la configuración central.
```

---

# 7. Fase 2 — S00 Raw Data Preparation

## Objetivo

Consolidar los archivos crudos sin filtrar sesiones ni aplicar calendario.

## Entradas

```text
data/00_source/*.txt
```

## Acciones

1. inventariar archivos;
2. extraer contrato;
3. leer con tipos explícitos;
4. validar esquema;
5. validar timestamps;
6. concatenar;
7. ordenar;
8. detectar duplicados;
9. detectar superposiciones;
10. documentar gaps entre archivos;
11. guardar Parquet crudo;
12. guardar manifest.

## Validaciones obligatorias

- columnas esperadas;
- OHLC coherente;
- volumen no negativo;
- precios positivos;
- timestamps parseables;
- duplicados exactos;
- timestamps duplicados;
- conflictos entre contratos;
- cobertura por archivo;
- checksum de archivos fuente.

## Salidas

```text
data/01_raw/mnq_raw.parquet
data/01_raw/mnq_raw_summary.json
manifests/s00_source_manifest.csv
reports/stage_reports/S00_report.md
```

## Criterio de aprobación

- cada archivo fuente está inventariado;
- ninguna transformación temporal se aplica;
- timezone sigue documentado como no confirmado si no existe evidencia;
- todo gap queda explicado o marcado como pendiente.

## Estado: APROBADO (S00 v2)

Implementado con sufijo `_v2` para no sobrescribir nombres oficiales previos
mientras no existía artefacto real que proteger:

```text
data/01_raw/mnq_raw_v2.parquet
data/01_raw/mnq_raw_v2_summary.json
data/01_raw/mnq_raw_v2_manifest.json   (autoritativo para hashes/staleness)
data/01_raw/mnq_raw_v2_gaps.parquet    (catálogo completo de gaps)
manifests/s00_source_manifest.csv      (vista derivada, no autoritativa)
reports/stage_reports/S00_v2_report.md
```

Resultado: 2.172.640 filas, 0 rechazadas, 35/35 pruebas aprobadas (25
unitarias + 10 de integración sobre el corpus real). Criterio de aprobación
cumplido: los 26 archivos fuente están inventariados con hash SHA-256;
ninguna transformación temporal se aplicó (tz-naive, sin `tz_localize` ni
`tz_convert`); `timezone_assumption=UTC` queda documentado explícitamente
como no confirmado; todos los gaps quedan catalogados en
`mnq_raw_v2_gaps.parquet`, y los dos casos extraordinarios (gap interno
MNQM23 y gap de transición H25→M25) quedan marcados `no_resuelto`, no
resueltos por conveniencia. `notebooks/S00_raw_data_preparation.ipynb` (v1)
no fue modificada y se conserva como evidencia histórica. Detalle completo
en `01_CURRENT_DECISIONS.md §31` y `02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.1-bis`.

---

# 8. Fase 3 — S01 Intraday Data Preparation

## Objetivo

Construir el dataset intradía oficial.

## Decisiones previas obligatorias

- timezone de origen confirmada;
- calendario CME o regla explícita;
- horario 04:30–16:00;
- política de cierres anticipados;
- regímenes vigentes;
- política de rollover y secuencias.

## Acciones

1. localizar timezone de origen;
2. convertir a America/New_York;
3. asignar fecha operativa;
4. aplicar calendario;
5. filtrar horario;
6. auditar frecuencia;
7. clasificar jornadas;
8. separar sesiones válidas, especiales e incompletas;
9. asignar `minute_of_day`;
10. asignar `regime_id`;
11. validar contratos;
12. guardar dataset.

## No permitido

- eliminar toda jornada con menos de 691 barras sin análisis;
- asignar régimen mediante valor por defecto;
- cargar un artefacto viejo sin comprobar versión;
- aplicar calendario NASDAQ.

## Salidas

```text
data/02_intraday/mnq_intraday.parquet
data/02_intraday/mnq_intraday_summary.json
data/02_intraday/trading_day_audit.parquet
data/02_intraday/regime_distribution.parquet
reports/stage_reports/S01_report.md
```

## Validaciones obligatorias

- índice único y ordenado;
- timezone correcta;
- límites horarios;
- coherencia `date`;
- frecuencia;
- gaps;
- OHLC;
- volumen;
- contratos;
- distribución de regímenes;
- barra 16:00;
- días especiales;
- comparación contra fuente.

## Puerta de aceptación

Claude debe producir un resumen de:

```text
días incluidos
días excluidos
días especiales
gaps
cobertura anual
barras por régimen
transiciones de contrato
```

---

# 9. Fase 4 — Auditoría de rollover

## Objetivo

Determinar si los cambios de contrato contaminan el pipeline.

## Acciones

Para cada transición:

1. identificar contrato saliente;
2. identificar contrato entrante;
3. localizar timestamps;
4. calcular gap temporal;
5. calcular gap de precio;
6. comprobar cambio dentro o fuera de sesión;
7. medir ventanas afectadas;
8. medir targets potencialmente afectados;
9. decidir tratamiento.

## Opciones de tratamiento

```text
excluir ventanas
reiniciar rolling por contrato
ajustar serie
mantener sin ajuste
```

## Salidas

```text
data/02_intraday/roll_audit.parquet
reports/stage_reports/ROLL_AUDIT_report.md
```

## Puerta de aceptación

No avanzar a features o targets sensibles a niveles sin una decisión explícita.

---

# 10. Fase 5 — S02 Intraday Data Analysis

## Objetivo

Comprender el dataset oficial reconstruido.

## Análisis principales

- cobertura;
- retornos;
- rangos;
- volumen;
- estructura de vela;
- estacionalidad intradía;
- comportamiento por régimen;
- estabilidad anual;
- ventanas 30/60/90;
- autocorrelación;
- heterocedasticidad;
- outliers;
- efectos de rollover;
- distribución de sesiones especiales.

## Reglas

- ninguna variable futura se usa como feature;
- los resultados por régimen usan la definición vigente;
- toda normalización se documenta;
- “cambio estructural” se reserva para pruebas formales.

## Salidas

```text
reports/stage_reports/S02_report.md
data/02_intraday/s02_summary.parquet
```

## Criterio de aprobación

Debe quedar justificado si 30/60/90 continúan como horizontes candidatos.

---

# 11. Fase 6 — Diseño de validación temporal

## Objetivo

Congelar folds antes de seleccionar targets y features.

## Folds base

```text
WF_01:
Train general: 2020–2021
Evaluación externa: 2022

WF_02:
Train general: 2020–2022
Evaluación externa: 2023

WF_03:
Train general: 2020–2023
Evaluación externa: 2024
```

## Validación interna

Dentro de cada train general:

```text
último bloque cronológico
```

Preferencia inicial:

```text
último trimestre del último año de train
```

Debe verificarse que el tamaño sea suficiente.

## Estado de años

```text
2025 = OOS histórica de desarrollo
2026 = test final parcial potencial
```

## Reglas

- los folds deben guardarse en config;
- ningún notebook redefine los años;
- purging se determina según intervalos reales;
- no se usa evaluación externa para tuning;
- 2026 no se consulta antes de congelar decisiones.

## Salidas

```text
config/folds_config.yaml
data/validation/folds.parquet
reports/stage_reports/VALIDATION_DESIGN.md
```

---

# 12. Fase 7 — S03 Threshold Calibration

## Objetivo

Recalcular thresholds sin contaminar evaluaciones futuras.

## Diseño

Para cada fold:

```text
thresholds ajustados exclusivamente con train
```

Comparaciones:

```text
global
por régimen
pooled
mediana anual
```

Candidatos:

```text
p40
p50
p60
```

Horizontes:

```text
30
60
90
```

## Decisión pendiente obligatoria

Elegir si las excursiones se definen con:

```text
close
high/low
ambas como targets separados
```

No mezclar semánticas.

## Salidas

```text
data/03_thresholds/fold_thresholds.parquet
data/03_thresholds/global_benchmark.parquet
reports/stage_reports/S03_report.md
```

## Puerta de aceptación

Claude debe mostrar:

- estabilidad por fold;
- diferencias por régimen;
- cantidad de muestras;
- sensibilidad a percentiles;
- efecto de sesiones especiales;
- efecto de rollover.

---

# 13. Fase 8 — S04 Target Design

## Objetivo

Rediseñar y comparar targets antes de fijar uno definitivo.

## Familias mínimas

```text
DIR
BAR
OPC histórico
targets continuos
targets binarios alternativos
targets jerárquicos
```

## OPC histórico

Se conserva solo como benchmark:

```text
opc_p50_h60_tp15_sl10
```

Mapping:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

## Alternativas a evaluar

### Alternativa A — Dirección y barrera separadas

```text
modelo 1: dirección
modelo 2: operar/no operar
modelo 3: TP/SL condicionado
```

### Alternativa B — Target continuo

```text
retorno futuro
excursión
utility score
```

### Alternativa C — Clasificación binaria

```text
long vs no-long
short vs no-short
trade vs no-trade
```

### Alternativa D — Clase ambigua

Separar casos donde OHLCV no permite determinar orden intrabar.

## Validaciones

- mapeo único;
- clases permitidas;
- causalidad;
- no cruzar días;
- no cruzar contratos;
- distribución por fold;
- estabilidad temporal;
- semántica de entrada;
- relación con ejecución.

## Salidas

```text
data/04_targets/
config/target_config.yaml
reports/stage_reports/S04_report.md
```

## Puerta de aceptación

No elegir un target únicamente por balance o tasa de eventos.

Debe evaluarse:

- significado operativo;
- predictibilidad;
- estabilidad;
- compatibilidad con ejecución;
- desbalance;
- calibración futura.

---

# 14. Fase 9 — S05 Feature Engineering

## Objetivo

Construir una biblioteca causal de features.

## Familias

- barra actual;
- retornos;
- momentum;
- rango;
- volatilidad;
- posición relativa;
- volumen;
- contexto temporal;
- indicadores técnicos;
- features de régimen;
- alternativas de normalización.

## Conjuntos iniciales

```text
BASE_RELATIVE
BASE_WITH_LEVELS
COMPACT_DOMAIN
FULL_CAUSAL
```

## Reglas

- cálculo dentro de jornada;
- reinicio por contrato cuando corresponda;
- no usar target;
- no usar metadata prohibida;
- preprocesamiento dentro del fold;
- selección dentro del train;
- versionado de nombres y fórmulas.

## Selección de features

Comparar dentro del train:

- manual;
- correlación;
- Mutual Information;
- regularización;
- permutation importance;
- SHAP interno;
- clustering;
- PCA.

## Salidas

```text
data/05_features/feature_matrix.parquet
data/05_features/feature_metadata.parquet
config/feature_config.yaml
reports/stage_reports/S05_report.md
```

## Puerta de aceptación

Cada feature debe incluir metadata:

```text
nombre
familia
fórmula
lookback
causalidad
unidad
usa nivel
usa volumen
cruza sesión
requiere escalado
```

---

# 15. Fase 10 — S06 Tabular Baselines

## Objetivo

Establecer el nivel mínimo que deben superar modelos complejos.

## Modelos obligatorios

```text
Dummy
Logistic Regression
Decision Tree
Random Forest
HistGradientBoosting
LightGBM o XGBoost
```

## Reglas

- mismos timestamps;
- mismos folds;
- mismas features;
- tuning interno;
- class weights como variante;
- resultados por clase;
- calibración separada;
- registrar todas las configuraciones.

## Métricas

```text
macro F1
balanced accuracy
log loss
accuracy
precision/recall/F1 por clase
matriz de confusión
```

## Salidas

```text
data/09_predictions/tabular/
data/10_metrics/tabular/
reports/model_reports/tabular_baselines.md
```

## Puerta de aceptación

Solo avanzar a redes si existe:

- baseline reproducible;
- evidencia de señal fuera de muestra;
- ausencia de errores de pipeline;
- comparación con priors;
- análisis de estabilidad.

---

# 16. Fase 11 — S07 Sequence Dataset

## Objetivo

Construir secuencias válidas y eficientes.

## Lookbacks

```text
30
60
90
```

Primario:

```text
60
```

## Forma lógica

```text
X: n_samples × lookback × n_features
y: n_samples
```

## Reglas

- una jornada;
- un contrato;
- sin gaps;
- dentro del fold;
- target alineado al último timestamp;
- preprocesamiento ajustado con train;
- mismos timestamps para todos los modelos.

## Implementación física

Preferir:

- generadores;
- chunks;
- memoria mapeada;
- `tf.data`;
- `PyTorch Dataset`.

Evitar crear múltiples copias completas.

## Validaciones

- shape;
- timestamps;
- orden;
- ausencia de futuro;
- continuidad;
- distribución de clases;
- correspondencia con dataset tabular.

## Salidas

```text
data/07_sequences/
reports/stage_reports/S07_report.md
```

---

# 17. Fase 12 — S08 Model Training

## Objetivo

Entrenar modelos bajo un protocolo común.

## Modelos

```text
MLP
CNN1D
GRU
LSTM
TCN
```

## Configuración base

- semilla definida;
- arquitectura pequeña;
- máximo de épocas;
- early stopping interno;
- batch size probado internamente;
- optimizer documentado;
- class weights como variante;
- checkpoint del mejor estado;
- repetición con varias semillas predefinidas.

## Reglas

- evaluación externa una sola vez por configuración congelada;
- no elegir semillas favorables;
- no usar P&L externo para tuning;
- no cambiar target desde una notebook de modelo.

## Salidas

```text
data/08_models/
data/09_predictions/
data/10_metrics/
reports/model_reports/
```

---

# 18. Fase 13 — S09 Model Comparison

## Objetivo

Comparar modelos sin ocultar inestabilidad.

## Comparaciones

- promedio de folds;
- desviación;
- peor fold;
- desempeño por clase;
- desempeño por régimen;
- desempeño por año;
- estabilidad de semillas;
- coste computacional;
- memoria;
- calibración preliminar.

## Regla de selección

Prioridad:

```text
robustez
→ generalización
→ calibración
→ simplicidad
→ coste
```

No seleccionar únicamente por mejor Macro-F1 promedio.

## Salidas

```text
reports/model_reports/model_comparison.md
data/10_metrics/model_comparison.parquet
```

---

# 19. Fase 14 — S10 Probability Calibration

## Objetivo

Evaluar si las probabilidades pueden interpretarse operativamente.

## Métodos candidatos

```text
Platt scaling
isotonic regression
temperature scaling
Dirichlet calibration
```

Solo dentro de train mediante predicciones internas u OOF.

## Métricas

```text
log loss
Brier Score
curvas de calibración
ECE
coverage
confidence vs accuracy
```

## Regla

No utilizar thresholds de confianza ni position sizing antes de esta etapa.

---

# 20. Fase 15 — S11 Backtesting

## Objetivo

Transformar predicciones congeladas en una simulación causal.

## Entradas

- predicciones OOS;
- probabilidades calibradas;
- timestamps;
- reglas operativas congeladas;
- precios crudos;
- costos.

## Reglas mínimas

- señal disponible después de cerrar barra `t`;
- entrada en evento posterior;
- comisión por lado;
- spread;
- slippage;
- rollover;
- duración;
- TP/SL;
- posición existente;
- señales superpuestas;
- cierre de sesión.

## Ambigüedad intrabar

Comparar:

```text
peor caso
exclusión
clase ambigua
datos granulares si se adquieren
```

## Métricas

```text
P&L bruto
P&L neto
Sharpe
Sortino
drawdown
Calmar
turnover
win rate
expectancy
resultados por fold
resultados por año
resultados por régimen
```

## Puerta de aceptación

Ningún resultado financiero se presenta sin:

- costos;
- reglas causales;
- sensibilidad;
- desglose temporal;
- comparación con benchmark.

---

# 21. Fase 16 — Test final 2026

## Objetivo

Evaluar una única vez el pipeline congelado.

## Condiciones

Antes de abrir 2026 deben quedar congelados:

- datos;
- targets;
- features;
- folds;
- modelo;
- hiperparámetros;
- calibración;
- reglas de decisión;
- backtest;
- costes.

## Limitación

```text
2026 termina el 17 de abril
```

Debe interpretarse como test parcial.

## Salidas

```text
reports/final_report/2026_final_test.md
```

---

# 22. Registro de experimentos

Cada fila del registro debe contener:

```text
experiment_id
timestamp
git_commit
data_version
target_version
feature_version
fold
model
hyperparameters
seed
train_period
internal_validation_period
external_evaluation_period
metrics
artifact_paths
status
notes
```

Estados:

```text
PLANNED
RUNNING
COMPLETED
FAILED
REJECTED
SELECTED
```

No se eliminan resultados negativos.

---

# 23. Criterios de congelamiento

Una decisión se considera congelada cuando:

1. está documentada;
2. tiene versión;
3. fue aprobada;
4. no depende del fold externo;
5. tiene impacto downstream identificado.

Cambiar una decisión congelada crea una nueva versión del pipeline.

Ejemplo:

```text
target_v1
feature_set_v1
folds_v1
model_protocol_v1
```

---

# 24. Estrategia de trabajo con Claude

## Para cada notebook

Claude debe recibir:

1. objetivo;
2. entradas;
3. salidas;
4. decisiones vigentes;
5. errores históricos relevantes;
6. validaciones obligatorias;
7. condiciones para detenerse.

## Formato de solicitud recomendado

```text
Desarrolla únicamente la notebook SXX.

Antes de escribir código:
- resume las decisiones relevantes;
- enumera riesgos;
- identifica entradas y salidas;
- señala contradicciones;
- no avances si falta una decisión upstream.

Después:
- propone estructura;
- implementa por bloques;
- agrega validaciones;
- genera manifest;
- redacta resumen de resultados.
```

## No pedir a Claude

```text
“Haz todo el proyecto de una vez.”
“Entrena todos los modelos y elige el mejor.”
“Optimiza para obtener el mayor P&L.”
```

Estas instrucciones aumentan el riesgo de decisiones implícitas y sobreajuste.

---

# 25. Orden práctico de ejecución

## Bloque A — Rehacer datos

```text
Fase 0
Fase 1
S00
S01
Rollover audit
S02
```

## Bloque B — Rehacer problema predictivo

```text
Validación
S03
S04
S05
```

## Bloque C — Baselines

```text
S06
```

## Bloque D — Secuencias y modelos

```text
S07
S08
S09
S10
```

## Bloque E — Trading

```text
S11
Test 2026
```

---

# 26. Entregables mínimos por stage

Cada stage debe entregar:

```text
notebook ejecutada
código reutilizable en src/
configuración
artefactos
manifest
reporte Markdown
validaciones
estado de aprobación
```

---

# 27. Checklist antes de comenzar

- [ ] Contextos cargados en Claude.
- [ ] Decisiones vigentes leídas.
- [ ] Problemas conocidos leídos.
- [ ] Plan leído.
- [ ] Archivos fuente disponibles.
- [ ] Timezone confirmada o marcada pendiente.
- [ ] Calendario definido.
- [ ] Regímenes aprobados.
- [ ] Repositorio limpio.
- [ ] Versionado activado.
- [ ] Rutas relativas.
- [ ] Registro de experimentos creado.
- [ ] 2026 bloqueado.

---

# 28. Checklist antes de entrenar modelos

- [ ] Dataset intradía aprobado.
- [ ] Jornadas especiales auditadas.
- [ ] Rollover auditado.
- [ ] Folds congelados.
- [ ] Thresholds fold-aware.
- [ ] Target aprobado.
- [ ] Mapping verificado.
- [ ] Features causales.
- [ ] Selección dentro del train.
- [ ] Baselines completos.
- [ ] Métricas predefinidas.
- [ ] Class weights como variante.
- [ ] Artefactos versionados.

---

# 29. Checklist antes del backtest

- [ ] Predicciones estrictamente OOS.
- [ ] Probabilidades calibradas o declaradas no calibradas.
- [ ] Regla de señal congelada.
- [ ] Entrada causal.
- [ ] Costes definidos.
- [ ] Slippage definido.
- [ ] Rollover definido.
- [ ] Ambigüedad intrabar tratada.
- [ ] Posiciones simultáneas tratadas.
- [ ] Sensibilidad preparada.

---

# 30. Criterio de finalización

La reconstrucción se considerará completa cuando exista:

1. pipeline reproducible;
2. linaje de artefactos;
3. validación temporal;
4. modelos comparables;
5. calibración;
6. backtest causal;
7. test 2026 parcial;
8. reporte de limitaciones;
9. registro completo de experimentos;
10. instrucciones para actualización futura.

---

# 31. Conclusión

El proyecto no debe reiniciarse como una búsqueda abierta de modelos.

Debe reconstruirse como una cadena controlada:

```text
datos
→ definición del problema
→ validación
→ targets
→ features
→ baselines
→ modelos
→ calibración
→ backtesting
→ test final
```

La prioridad no es llegar rápidamente a una red neuronal.

La prioridad es garantizar que cualquier resultado positivo provenga de un proceso:

- causal;
- reproducible;
- temporalmente válido;
- libre de errores upstream conocidos;
- separado del backtesting oportunista.
