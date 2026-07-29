# 01 — CURRENT DECISIONS

## 1. Propósito y autoridad del documento

Este archivo define las decisiones que deben considerarse **vigentes** al reconstruir y continuar el proyecto de Machine Learning aplicado al futuro **Micro E-mini Nasdaq-100 (MNQ)**.

Debe utilizarse como fuente normativa para Claude. Su función es evitar que decisiones históricas, resultados exploratorios o configuraciones provisionales sean interpretados como reglas definitivas.

Jerarquía de interpretación:

```text
01_CURRENT_DECISIONS.md
→ 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
→ 00_MNQ_MASTER_CONTEXT.md
→ archivos S00–S07_CONTEXT
→ notas de estudio de los libros
→ libros completos
```

Cuando un archivo histórico contradiga este documento, prevalece este documento.

Las decisiones aquí registradas pueden modificarse, pero cualquier cambio debe:

1. quedar documentado;
2. indicar su fundamento;
3. identificar qué artefactos quedan afectados;
4. obligar a regenerar las etapas dependientes cuando corresponda.

---

## 2. Decisión general sobre el proyecto

El proyecto puede reconstruirse desde cero con Claude.

La reconstrucción implica:

```text
reiniciar la implementación
≠
descartar el conocimiento acumulado
```

Los stages S00–S07 existentes se conservarán como:

- documentación del trabajo realizado;
- evidencia de resultados exploratorios;
- registro de problemas metodológicos;
- referencia para evitar repetir errores;
- fuente de comparaciones históricas.

No se continuará automáticamente desde `S07_01`. Primero se revisarán y congelarán las decisiones de datos, calendario, regímenes, targets, features y validación.

---

## 3. Objetivo vigente

El objetivo es desarrollar un pipeline reproducible para estudiar si datos intradía de MNQ contienen señal predictiva utilizable por modelos de Machine Learning.

El proyecto debe evaluar de forma separada:

```text
1. capacidad predictiva;
2. calidad probabilística;
3. utilidad operativa;
4. rendimiento financiero neto;
5. estabilidad temporal.
```

El éxito no se definirá por una única métrica ni por un único modelo.

Una mejora estadística no se interpretará automáticamente como una estrategia rentable.

---

## 4. Alcance de los datos

### Instrumento principal

```text
Micro E-mini Nasdaq-100 Futures
Símbolo: MNQ
Mercado: CME Globex
```

### Fuente histórica disponible

```text
NinjaTrader
Barras OHLCV de 1 minuto
Contratos trimestrales
Periodo aproximado: 2020–2026
```

### Datos principales disponibles

```text
open
high
low
close
volume
contract
timestamp
```

El proyecto actual utiliza un único instrumento y no dispone de:

- ticks;
- bid y ask;
- spread observado;
- profundidad de mercado;
- dirección agresora;
- secuencia intraminuto;
- VWAP real;
- fills parciales.

Las técnicas que requieren esos datos no deben simularse como si fueran observables.

---

## 5. Zona horaria y sesión operativa

La zona horaria operativa es:

```text
America/New_York
```

El horario intradía objetivo es:

```text
04:30–16:00
```

Los extremos se consideran incluidos mientras la auditoría de datos confirme que esa convención es consistente con la fuente.

Antes de aplicar calendarios o asignar la fecha operativa, los timestamps deben:

```text
localizarse correctamente en la zona horaria de origen
→ convertirse a America/New_York
→ asignarse a la jornada operativa
```

La interpretación histórica de los timestamps como UTC debe confirmarse documentalmente con la fuente o la configuración de exportación.

---

## 6. Regímenes intradía vigentes

La convención de trabajo es:

| regime_id | Régimen | Horario New York |
|---:|---|---|
| 0 | Overnight | 04:30–08:29 |
| 1 | Pre-market | 08:30–09:29 |
| 2 | Opening | 09:30–10:29 |
| 3 | Regular | 10:30–14:59 |
| 4 | Closing | 15:00–16:00 |

Reglas:

- la barra de las 16:00 pertenece a `Closing`;
- `Regular` termina a las 14:59;
- `Closing` comienza a las 15:00;
- ninguna barra puede quedar asignada por un valor por defecto silencioso;
- la distribución de barras por régimen debe validarse explícitamente.

Estos regímenes constituyen una **hipótesis de segmentación temporal**. No deben presentarse como regímenes económicos demostrados.

Deben compararse:

```text
modelo global sin régimen
modelo global con contexto temporal
modelos separados por régimen
```

---

## 7. Calendario y jornadas

No se utilizará automáticamente el calendario NASDAQ para MNQ.

Debe emplearse:

- un calendario aplicable a futuros de índices de CME; o
- una definición explícita y auditada de las jornadas operativas.

Las jornadas incompletas no se eliminarán únicamente por tener menos de 691 barras.

Cada jornada reducida debe clasificarse como:

```text
cierre anticipado legítimo
sesión especial
gap de datos
archivo incompleto
error de fuente
```

La política de inclusión o exclusión debe quedar congelada antes de reconstruir targets y features.

---

## 8. Contratos y rollover

La columna `contract` debe conservarse desde la ingestión.

Antes de construir el dataset definitivo se deben auditar todas las transiciones entre contratos:

- último timestamp del contrato saliente;
- primer timestamp del contrato entrante;
- gap temporal;
- gap de precio;
- cambio dentro o fuera de la jornada;
- ventanas históricas que atraviesen el cambio;
- targets futuros que atraviesen el cambio.

Decisión vigente:

```text
no ajustar toda la serie de forma automática
```

Primero se medirá la contaminación real.

Si se necesita una serie ajustada:

- se conservarán los precios crudos;
- los precios ajustados se utilizarán solo para los fines definidos;
- la ejecución y el P&L utilizarán precios realmente negociables;
- ninguna secuencia cruzará contratos salvo que exista una decisión explícita y validada.

---

## 9. Frecuencia y límites de las muestras

La frecuencia base continuará siendo un minuto.

Las barras de tiempo de un minuto son la estructura principal porque:

- son los datos disponibles;
- preservan el horario;
- permiten definir horizontes en minutos;
- mantienen el contexto intradía.

Las barras aproximadas de volumen, dólares o eventos podrán evaluarse posteriormente como experimentos separados.

No se presentarán como equivalentes a barras construidas desde trades individuales.

Las ventanas y targets no deben cruzar:

- jornadas;
- gaps;
- contratos, salvo decisión explícita;
- límites temporales de train, validación o test.

---

## 10. Horizontes temporales

Los horizontes candidatos principales se mantienen en:

```text
30 minutos
60 minutos
90 minutos
```

El horizonte de 60 minutos se conserva como configuración primaria de comparación porque históricamente mostró un equilibrio razonable entre movimiento y cantidad de muestras.

Esto no significa que 60 minutos sea definitivamente superior.

Los tres horizontes deben compararse bajo el mismo protocolo y con selección realizada dentro del train permitido.

---

## 11. Targets

Se conservarán como familias de estudio:

```text
DIR
BAR
OPC
targets continuos complementarios
```

### 11.1. DIR

DIR se mantiene como benchmark direccional.

Debe revisarse su definición para asegurar que:

- la etiqueta corresponda exactamente al horizonte;
- los thresholds se estimen solo con datos de entrenamiento;
- el tratamiento de empates sea explícito;
- los resultados sean comparables entre folds.

### 11.2. BAR

BAR se mantiene como benchmark estadístico para estudiar alcanzabilidad de barreras.

No se considerará una señal desplegable independiente mientras la dirección sea seleccionada retrospectivamente mediante información futura.

La tasa histórica de TP no se interpretará como win rate.

### 11.3. OPC

El target histórico prioritario es:

```text
opc_p50_h60_tp15_sl10
```

Su codificación real histórica es:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

Este mapping debe leerse desde metadata y validarse contra los datos. No debe escribirse manualmente en cada notebook.

OPC se conserva como candidato operativo principal para comparación histórica, pero **no se considera todavía un target definitivo**.

Antes de congelarlo deben revisarse:

- selección retrospectiva de dirección;
- uso exclusivo de cierres para evaluar barreras;
- heterogeneidad de `NO_TRADE`;
- desbalance severo;
- coherencia entre threshold, entrada y ejecución;
- alternativas de formulación.

### 11.4. Targets continuos

La reconstrucción no debe limitarse a OPC multiclase.

También podrán estudiarse:

- retorno futuro;
- excursión máxima;
- magnitud del movimiento;
- volatilidad futura;
- tiempo hasta evento;
- formulaciones binarias o jerárquicas.

Cada target debe evaluarse por su significado predictivo y operativo.

---

## 12. Thresholds y barreras

Los candidatos históricos se mantienen como punto de partida:

```text
percentiles: p40, p50, p60
horizontes: 30, 60, 90
threshold por régimen
threshold global como benchmark
```

El escenario base histórico es:

```text
p50 + h60
```

Los thresholds deben recalcularse dentro del train correspondiente cuando participen en la definición de labels evaluados fuera de muestra.

Debe definirse explícitamente si las excursiones y barreras se calculan con:

```text
close
o
high/low
```

No deben mezclarse sin documentar que representan fenómenos diferentes.

La configuración histórica:

```text
TP = 1,5 × threshold
SL = 1,0 × threshold
```

se conservará como benchmark, no como regla operativa definitiva.

---

## 13. Features

Todas las features deben ser causales:

```text
X_t = información disponible hasta t
```

Reglas obligatorias:

- ningún target o auxiliar futuro puede usarse como predictor;
- rolling y lags se calculan únicamente con datos anteriores o contemporáneos;
- las transformaciones no cruzan jornadas ni contratos prohibidos;
- escalado, imputación, winsorización, PCA y selección se ajustan solo con train;
- la selección supervisada se ejecuta dentro de cada fold o dentro de un protocolo interno equivalente.

Familias candidatas:

- retornos y lags;
- estructura de vela;
- rangos;
- volatilidad;
- momentum;
- posición relativa;
- volumen;
- contexto temporal;
- indicadores técnicos causales.

Se compararán al menos:

```text
features con niveles
features relativas o normalizadas
features compactas
features completas
```

Las variables de nivel absoluto deben tratarse con cautela porque pueden codificar año, contrato o tendencia secular.

---

## 14. Metadata y variables de contexto

Las siguientes columnas se tratarán principalmente como metadata:

```text
date
year
dataset_split
contract
fold_id
validation_year
target_name
dataset_name
```

Reglas:

- `dataset_split` y `fold_id` están prohibidas como features;
- `year` no se utilizará como predictor por defecto;
- `contract` no se utilizará como predictor por defecto;
- cualquier uso de año o contrato requerirá una hipótesis y una prueba específica.

Contexto temporal candidato:

```text
minute_of_day
representación cíclica del tiempo
day_of_week
regime_id
```

`regime_id` debe codificarse categóricamente, no como una magnitud ordinal automática.

---

## 15. Selección y reducción de features

La selección histórica realizada sobre todo 2020–2024 no se considerará evidencia walk-forward estrictamente independiente.

En la reconstrucción, la selección deberá:

```text
ajustarse dentro del train de cada fold
o
congelarse antes de observar las evaluaciones externas y declararse como hipótesis previa
```

Se podrán comparar:

- todas las features;
- selección manual por dominio;
- eliminación de redundancia;
- selección por Mutual Information;
- modelos regularizados;
- clustering de features;
- PCA global o por familias;
- importancia multivariada.

Ningún método de importancia demostrará causalidad por sí mismo.

---

## 16. Validación temporal

No se utilizarán splits aleatorios para afirmar generalización futura.

Diseño walk-forward base:

```text
WF_01
Train general: 2020–2021
Evaluación externa: 2022

WF_02
Train general: 2020–2022
Evaluación externa: 2023

WF_03
Train general: 2020–2023
Evaluación externa: 2024
```

Dentro de cada train general debe existir una validación interna temporal para:

- hiperparámetros;
- early stopping;
- selección de arquitectura;
- calibración;
- selección de features;
- thresholds operativos;
- class weights;
- reglas de decisión.

La evaluación externa no se utilizará para modificar el modelo correspondiente.

Purging y embargo se aplicarán cuando los intervalos reales de las etiquetas demuestren solapamiento o contaminación. No se incorporarán mecánicamente sin revisar los tiempos de cada muestra.

---

## 17. Estado de 2025 y 2026

### 2025

```text
2025 ya fue utilizado extensamente.
```

Por lo tanto:

- no es un holdout ciego;
- puede conservarse como evaluación OOS histórica de desarrollo;
- no debe presentarse como test final independiente.

### 2026

```text
2026 llega hasta el 17 de abril.
```

Puede reservarse como evaluación final parcial únicamente si:

- no se utiliza para seleccionar targets;
- no se utiliza para seleccionar features;
- no se utiliza para tuning;
- no se utiliza para elegir reglas operativas;
- todas las decisiones se congelan antes de consultarlo.

La limitada extensión de 2026 debe declararse en toda interpretación.

---

## 18. Baselines obligatorios

Todo modelo complejo debe compararse contra baselines adecuados.

Baselines mínimos:

```text
predicción de clase mayoritaria
probabilidades según frecuencia del train
regla simple de retorno anterior
momentum simple
reversión simple
Logistic Regression
árbol individual
```

Para regresión:

```text
media o mediana del train
modelo lineal regularizado
```

Los baselines deben usar los mismos timestamps y el mismo conjunto de evaluación que los modelos comparados.

---

## 19. Modelos candidatos

### Tabulares

```text
Logistic Regression
Ridge / Lasso / Elastic Net cuando corresponda
Decision Tree
Random Forest
HistGradientBoosting
LightGBM
XGBoost
CatBoost
MLP tabular
```

### Secuenciales

```text
CNN1D
GRU
LSTM
TCN
```

La comparación debe comenzar con arquitecturas pequeñas y controladas.

No se asumirá que Deep Learning supera a boosting ni que un ensamble supera a modelos individuales.

Los modelos secuenciales utilizarán una forma lógica común:

```text
muestras × pasos temporales × canales
```

El MLP podrá utilizar una representación tabular o una ventana aplanada, pero no se describirá como modelo temporal explícito.

---

## 20. Secuencias

Una secuencia válida debe:

- terminar en el timestamp del target;
- contener únicamente información disponible hasta ese instante;
- pertenecer a una única jornada;
- no contener gaps;
- pertenecer a un único contrato;
- respetar los límites del fold;
- conservar el orden temporal.

Lookbacks candidatos:

```text
30
60
90
```

El lookback del modelo y la historia efectiva de las features deben documentarse por separado.

La construcción física debe evitar duplicaciones de memoria innecesarias y podrá utilizar:

- generadores;
- datasets por chunks;
- memory mapping;
- `tf.data`;
- `PyTorch Dataset`;
- Zarr o HDF5 cuando estén justificados.

---

## 21. Desbalance de clases

El desbalance no se resolverá automáticamente mediante `class_weight="balanced"`.

Se compararán de forma controlada:

```text
sin pesos
class weights
focal loss
muestreo dentro del train
formulaciones alternativas del target
modelos jerárquicos
```

Cualquier peso, resampling o técnica de balanceo debe ajustarse únicamente con el train.

El efecto debe evaluarse por clase para evitar sobrepredicción de clases minoritarias.

---

## 22. Métricas predictivas

Para clasificación, la métrica principal actual es:

```text
macro F1
```

Métricas obligatorias adicionales:

```text
balanced accuracy
log loss
accuracy
precision por clase
recall por clase
F1 por clase
support
matriz de confusión
distribución de predicciones
```

La selección no se realizará solo por el promedio.

También se analizarán:

- desviación entre folds;
- peor fold;
- resultados por año;
- resultados por régimen;
- colapso hacia una clase;
- estabilidad entre semillas.

Para regresión:

```text
MAE
RMSE
correlación fuera de muestra
IC temporal cuando corresponda
```

---

## 23. Probabilidades y calibración

Las probabilidades de:

- regresión logística;
- Random Forest;
- boosting;
- softmax;
- sigmoide;

no se asumirán calibradas.

Antes de utilizar confianza para filtrar operaciones o dimensionar posiciones se deberán analizar:

```text
log loss
Brier Score
curvas de calibración
calibración por clase
cobertura
estabilidad entre folds y años
```

Los calibradores se ajustarán dentro del train mediante predicciones internas u out-of-fold.

---

## 24. Evaluación financiera

La evaluación financiera será una etapa separada del entrenamiento predictivo.

Debe reportar:

```text
P&L bruto
P&L neto
comisiones
tasas
spread
slippage
turnover
Sharpe
Sortino
máximo drawdown
Calmar
resultados por operación
resultados por sesión
resultados por día
resultados por fold
resultados por año
resultados por régimen
```

Las métricas anualizadas se calcularán preferentemente con retornos diarios o por sesión, no anualizando mecánicamente retornos por minuto.

El rendimiento deberá compararse contra benchmarks operativos compatibles.

---

## 25. Causalidad de señal y ejecución

Si una feature necesita el cierre, high, low o volumen completo de la barra `t`, la señal solo existe después de finalizar esa barra.

La ejecución causal debe ocurrir en un evento posterior.

No se asumirá entrada al mismo `close_t` utilizado para calcular la señal.

Las reglas de entrada, TP, SL y salida deben definir:

- primer precio ejecutable;
- slippage;
- costes por lado;
- tratamiento de gaps;
- tratamiento de rollover;
- duración máxima;
- señales simultáneas;
- posiciones existentes.

---

## 26. Ambigüedad intrabar

Con OHLCV de un minuto no puede determinarse cuál ocurrió primero cuando una misma vela toca TP y SL.

La reconstrucción debe:

1. medir la frecuencia de casos ambiguos;
2. definir una política antes del backtest;
3. presentar análisis de sensibilidad.

Políticas candidatas:

```text
peor caso
mejor caso
exclusión
clase ambigua
resolución con datos más granulares
```

La política principal debe ser conservadora.

---

## 27. Reproducibilidad

Cada experimento debe registrar:

- identificador;
- fecha;
- versión del código;
- versión de datos;
- target;
- features;
- fold;
- modelo;
- hiperparámetros;
- semilla;
- preprocesamiento;
- métricas;
- predicciones;
- artefactos;
- resultado favorable o desfavorable.

No se eliminarán ni ocultarán experimentos negativos.

Las rutas guardadas en metadata deben ser relativas al proyecto cuando sea posible.

Cada artefacto generado debe identificar qué versiones upstream lo produjeron.

---

## 28. Uso del conocimiento de los libros

Las notas de Jansen, López de Prado y Murphy se utilizarán para:

- auditar decisiones;
- proponer experimentos;
- identificar riesgos;
- comparar metodologías;
- mejorar targets, features, modelos y validación.

No deben utilizarse para copiar implementaciones sin adaptación.

Claude debe distinguir en las notas:

```text
conocimiento explícito del autor
implicación validada para MNQ
propuesta experimental
decisión pendiente
contenido no transferible
```

Una propuesta de un capítulo no se convierte automáticamente en decisión del proyecto.

---

## 29. Decisiones que no están vigentes

No deben tratarse como vigentes:

- calendario NASDAQ aplicado a MNQ;
- regímenes históricos `Regular 10:30–15:29` y `Closing 15:30–15:59`;
- barra de 16:00 asignada a Overnight;
- eliminación automática de toda jornada con menos de 691 barras;
- mapping OPC provisional de S07;
- 2025–2026 como holdout completamente ciego;
- `class_weight` obligatorio;
- selección de features histórica como evidencia fold-independent;
- BAR como estrategia operativa;
- confianza del modelo como filtro de trading sin calibración;
- niveles absolutos como features válidas por defecto;
- continuar directamente con S07_01 sin auditoría upstream;
- usar el P&L externo para elegir reiteradamente configuraciones.

---

## 30. Condiciones para avanzar al entrenamiento definitivo

Antes de entrenar modelos definitivos deben quedar cerrados:

1. zona horaria confirmada;
2. calendario de MNQ definido;
3. política para jornadas especiales;
4. regímenes corregidos;
5. auditoría de rollover;
6. definición de targets revisada;
7. mapping de clases verificado;
8. features y metadata separadas;
9. selección dentro del train;
10. folds y validación interna congelados;
11. estado de 2025 y 2026 documentado;
12. baselines definidos;
13. métricas predefinidas;
14. registro de experimentos operativo;
15. reglas para no reutilizar evaluación externa.

Solo después debe reconstruirse Stage 07 y comenzar la comparación definitiva de modelos.

---

## 31. Estado vigente de S00 (APROBADO — S00 v2)

S00 fue reconstruido y aprobado formalmente. Esta sección fija cuál es el
artefacto de entrada vigente para cualquier trabajo posterior (incluido el
diseño de S01).

```text
Artefacto crudo vigente:   data/01_raw/mnq_raw_v2.parquet
Manifiesto autoritativo:   data/01_raw/mnq_raw_v2_manifest.json
Summary:                    data/01_raw/mnq_raw_v2_summary.json
Catálogo de gaps:           data/01_raw/mnq_raw_v2_gaps.parquet
Implementación:              src/data/s00_raw_ingestion.py
Notebook vigente:            notebooks/S00_raw_data_preparation_v2.ipynb
Pruebas:                     35/35 aprobadas (25 unitarias + 10 de integración)
Filas:                       2.172.640
Filas rechazadas:            0
```

`data/01_raw/mnq_raw.parquet` (nombre histórico, sin sufijo) **no existe y
no debe asumirse como artefacto vigente.** `notebooks/S00_raw_data_preparation.ipynb`
(v1, sin sufijo) **no fue modificada** y se conserva únicamente como
evidencia histórica — no debe ejecutarse como parte del pipeline vigente.

Decisiones confirmadas y vigentes sobre el dataset crudo:

```text
timestamps: tz-naive (sin conversión de zona horaria en S00)
timezone_assumption: UTC — INFERIDO, NO CONFIRMADO DOCUMENTALMENTE
timestamp_semantics (inicio vs. cierre de barra): NO CONFIRMADO
price_type: "Last" — inferido solo del nombre de archivo, sin confirmación del proveedor
```

Pendientes que **no bloquean** la aprobación de S00, pero sí deben
resolverse antes de tratar el dataset como completamente auditado:

```text
Gap interno MNQM23 (2023-04-05 18:03 → 2023-04-16 14:18, ~260h15min): NO_RESUELTO
Gap de transición H25→M25 (2025-03-21 13:30 → 2025-04-06 08:42, ~15d19h12min): NO_RESUELTO
Confirmación documental de zona horaria de origen: PENDIENTE
Chequeo automatizado de solapamiento de intervalos entre archivos: MEJORA MENOR PENDIENTE
  (verificado manualmente en la auditoría — 0 solapamientos — pero no
  automatizado dentro de s00_raw_ingestion.py)
```

Detalle completo: `reports/stage_reports/S00_v2_report.md` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.1-bis`.

---

## 32. Estado vigente de S01 (APROBADO — S01 v2)

S01 fue reconstruido y aprobado formalmente. Fija el artefacto de entrada
vigente para cualquier trabajo posterior (S02 en adelante).

```text
Artefacto intradía vigente:  data/02_intraday/mnq_intraday_v2.parquet
Auditoría de jornadas:       data/02_intraday/trading_day_audit_v2.parquet
Distribución por régimen:    data/02_intraday/regime_distribution_v2.parquet
Manifiesto autoritativo:     data/02_intraday/mnq_intraday_v2_manifest.json
Validación de zona horaria:  data/02_intraday/tz_validation_v2.json
Implementación:               src/data/s01_intraday_preparation.py
Config:                        config/intraday_config.yaml
Notebook vigente:              notebooks/S01_intraday_data_preparation_v2.ipynb
Pruebas:                       42/42 aprobadas (26 unitarias + 16 de integración)
Filas:                         1.087.777  (subconjunto full_coverage: 1.482 × 691 = 1.024.062, idéntico a S01 v1)
```

`data/02_intraday/mnq_intraday.parquet` (nombre histórico, sin sufijo)
**no debe asumirse como artefacto vigente.**
`notebooks/S01_intraday_data_preparation.ipynb` (v1, sin sufijo) **no fue
modificada** y se conserva únicamente como evidencia histórica — sus
resultados (regímenes con límites incorrectos, calendario NASDAQ, 80 días
eliminados sin clasificar) **no deben presentarse como vigentes.**

Decisiones confirmadas y vigentes sobre el dataset intradía:

```text
zona horaria de origen: UTC, seleccionada por comparación empírica
  programática de 3 hipótesis (UTC, America/New_York, America/Chicago)
  contra apertura 09:30 ET, cierre 16:00 ET y corte de mantenimiento CME
  (~17:00-18:00 ET) — NO asumida, NO confirmada documentalmente por el
  proveedor (timezone_provider_confirmation: false)
timestamp_semantics: unknown_not_confirmed — ninguna barra fue desplazada
ventana operativa: 04:30-16:00 America/New_York, ambos extremos incluidos,
  691 minutos esperados — decisión operativa CONVENCIONAL, explícitamente
  NO identificada como quiebre estructural óptimo (ver auditoría S01 v2)
regímenes: 5, mismos regime_id 0-4 y límites vigentes
  (10:30-14:59 Regular, 15:00-16:00 Closing, sin ruta default); único
  cambio: etiqueta de regime_id=0 pasa de "Overnight" a "Early_Premarket"
calendario: híbrido — CME_Equity como referencia + datos observados como
  evidencia principal; ningún día se excluye silenciosamente
jornadas parciales: conservadas y clasificadas explícitamente, nunca
  eliminadas solo por conteo de barras (ver desglose completo abajo)
```

Clasificación de las 2.309 fechas calendario del rango
(`trading_day_audit_v2.parquet`):

```text
1.482  full_coverage
  305  parciales CON datos: 244 partial_undetermined, 57 partial_early_close_cme,
       4 partial_gap_documented_s00
  522  fechas SIN datos: 475 no_data_weekend, 25 no_data_gap_documented_s00,
       11 no_data_cme_holiday, 11 no_data_undetermined
```

Los 29 días vinculados a gaps de S00 (12 a `s00_gap_M23`, 17 a
`s00_gap_H25_M25`) quedan explícitamente `is_model_eligible=false`, con
referencia trazable a `data/01_raw/mnq_raw_v2_gaps.parquet` — no se
rellenó, interpoló ni eliminó ningún dato.

Pendientes que **no bloquean** la aprobación de S01, pero deben
resolverse o revisarse antes de tratar el dataset como completamente
auditado:

```text
244 jornadas partial_undetermined + 11 no_data_undetermined: causa sin determinar
Patrón recurrente 16:20-16:30 (2019-2021): fuera de la ventana primaria,
  documentado, no afecta mnq_intraday_v2.parquet
Discrepancia CME_Equity vs "CME Globex Equity" (1 día, 2025-01-09): no revalidada
Confirmación documental de zona horaria de origen (heredada de S00): sigue pendiente
test_s00_integration.py::test_never_writes_to_productive_raw_dir: falla
  preexistente de S00 (asume data/01_raw/ vacío), no relacionada con S01 v2
```

Detalle completo: `reports/stage_reports/S01_v2_report.md` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.2-bis`.

---

## 33. Estado vigente de S02 (APROBADO — S02 v2, `APPROVED_WITH_KNOWN_LIMITATION`)

S02 fue reconstruido y aprobado formalmente, con una limitación técnica
conocida declarada explícitamente (no bloqueante). Fija el artefacto de
análisis exploratorio vigente para cualquier trabajo posterior (S03 en
adelante).

```text
Notebook oficial:      notebooks/S02_intraday_data_analysis_v2.ipynb
Módulo oficial:         src/data/s02_intraday_analysis.py
Config:                 config/s02_analysis_config.yaml
Reporte:                reports/stage_reports/S02_v2_report.md
Manifiesto autoritativo: data/02_intraday/s02_analysis_manifest.json
Pruebas:                26/26 unitarias + 12/12 integración (S02) —
                        114/115 en la suite completa del repo (única falla
                        preexistente de S00, no relacionada)
```

Artefactos oficiales generados (todos en `data/02_intraday/`, gitignorados
igual que el resto de `data/`):

```text
s02_summary.parquet
s02_coverage_summary.parquet
s02_regime_distribution.parquet
s02_ohlcv_stats_summary.parquet
s02_ohlcv_correlation.parquet
s02_window_validity_summary.parquet
s02_rollover_window_audit.parquet
s02_window_metrics_summary.parquet
s02_stability_summary.parquet
s02_rolling_summary.parquet
s02_temporal_instability_zones.parquet
s02_acf_summary.parquet
s02_dependence_tests_summary.parquet
```

`notebooks/S02_intraday_data_analysis.ipynb` (v1, sin sufijo) **no fue
modificada** y se conserva únicamente como evidencia histórica — sus
resultados (régimen contaminado, "Overnight" incluyendo la barra 16:00,
cobertura asumida 691 barras/día siempre, generador exploratorio de features,
"cambios estructurales" no formales, afirmación de holdout ciego 2025–2026)
**no deben presentarse como vigentes.**

Decisiones confirmadas y vigentes sobre el análisis intradía:

```text
poblaciones: all (cobertura/calidad/descriptivo) · full_day_eligible
  (cuantitativa principal) · partial_regime_eligible (secundaria, solo
  regímenes marcados regime_{id}_is_consecutive=True por S01) ·
  descriptive_only (solo descripción) · not_model_eligible (excluida de
  ventanas y métricas para stages posteriores)
ventanas: válidas solo si comparten simultáneamente fecha, segmento de
  minutos consecutivos (recalculado sobre la población filtrada), contrato
  único y consecutividad estricta de minuto a minuto; cambio de contrato:
  bloqueante, verificado empíricamente sin impacto (ver más abajo)
body_signed_pts / body_abs_pts: nombres inequívocos, "body_pts" a secas no
  se usa en ningún artefacto ni en el código
terminología: "inestabilidad temporal" / "cambio de distribución", nunca
  "cambio estructural" como conclusión; sin pruebas formales de ruptura
ACF y Ljung-Box: gap-aware (nunca forman un par entre observaciones de
  distinto consecutive_segment_id); ARCH-LM sigue sin serlo (ver limitación
  abajo)
targets: S02 no construye, evalúa ni selecciona DIR/BAR/OPC ni ningún otro
generador exploratorio de features histórico: no se ejecuta ni se reproduce
  en S02 v2; queda solo como antecedente en la notebook v1
```

Cifras verificadas por ejecución real (`s02_analysis_manifest.json` y
artefactos):

```text
Filas totales (población 'all'):        1.087.777
Jornadas full_day_eligible:              1.482  (691 barras/día exactas → 1.024.062 filas)
Filas full_day_eligible:                 1.024.062  (idéntico al histórico v1/S02 v1)
Filas partial_regime_eligible:              40.539  (119 días, solo regímenes consecutivos)
Transiciones de contrato detectadas:            25
Transiciones dentro de un mismo segmento:        0  (ningún rollover real cae dentro de
                                                     una jornada/segmento; el bloqueo por
                                                     contrato único no reduce validez alguna)
Validez de ventana full_day_eligible:    30m 91,46% · 60m 82,78% · 90m 74,10% (full)
Régimen full_day_eligible:               Early_Premarket 34,73% · Premarket 8,68% ·
                                          Opening 8,68% · Regular 39,07% · Closing 8,83%
```

**Limitación técnica conocida (no bloqueante):**

```text
ARCH-LM (statsmodels.stats.diagnostic.het_arch) todavía NO es gap-aware:
arma su matriz de regresores lageados sobre el array completo sin conocer
límites de consecutive_segment_id, la misma contaminación que tenía
Ljung-Box antes de corregirse en este cierre. Sus resultados son
únicamente diagnósticos exploratorios sobre heterocedasticidad condicional
y NO deben usarse como evidencia definitiva ni como criterio de selección
de features o modelos. Corregirlo (reimplementación manual, no es un simple
reescalado como la ACF) queda como tarea de mantenimiento metodológico
posterior, fuera del alcance de este cierre.
```

Pendientes heredados que **no bloquean** la aprobación de S02 (ya
documentados en S00/S01, S02 no los resuelve ni corresponde que lo haga):

```text
Zona horaria de origen y timestamp_semantics: siguen sin confirmación
  documental (heredado de S00/S01).
244 jornadas partial_undetermined + 11 no_data_undetermined: causa sin
  determinar (heredado de S01).
Auditoría general de rollover (Fase 4 del rebuild plan, transiciones crudas
  de contrato: gap de precio, contexto completo): sigue pendiente como
  etapa separada; S02 v2 solo aportó el impacto a nivel de ventana (0
  transiciones intra-segmento).
```

Detalle completo, incluyendo la validación focalizada que encontró y
corrigió la contaminación entre segmentos de ACF/Ljung-Box (y descartó dos
falsos positivos sobre validez de ventanas y sobre `Closing`/h=90):
`reports/stage_reports/S02_v2_report.md` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.3-bis`.
