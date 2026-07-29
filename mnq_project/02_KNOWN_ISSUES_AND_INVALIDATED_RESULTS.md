# 02 — KNOWN ISSUES AND INVALIDATED RESULTS

## 1. Propósito del documento

Este archivo registra los problemas conocidos del pipeline histórico MNQ y define qué resultados:

- permanecen válidos;
- son únicamente exploratorios;
- están metodológicamente comprometidos;
- deben regenerarse;
- no deben utilizarse como evidencia final.

Su objetivo es impedir que Claude reutilice resultados históricos sin considerar sus dependencias y limitaciones.

Este documento no afirma que todo el trabajo anterior sea inútil. La mayor parte conserva valor como:

- documentación;
- antecedente metodológico;
- evidencia exploratoria;
- diagnóstico de errores;
- guía para la reconstrucción.

Jerarquía documental:

```text
01_CURRENT_DECISIONS.md
→ 02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md
→ 00_MNQ_MASTER_CONTEXT.md
→ S00–S07_CONTEXT
→ notas de estudio
→ libros completos
```

Cuando un archivo histórico presente un resultado incompatible con este documento, debe interpretarse según el estado asignado aquí.

---

## 2. Clasificación de estados

Cada resultado o artefacto debe clasificarse en una de las siguientes categorías.

### 2.1. VÁLIDO

Puede conservarse y reutilizarse, sujeto a las condiciones documentadas.

### 2.2. VÁLIDO COMO ANTECEDENTE

Describe correctamente lo realizado, pero no constituye una decisión vigente ni evidencia final.

### 2.3. EXPLORATORIO

Puede orientar nuevas hipótesis, pero no debe utilizarse para afirmar generalización futura.

### 2.4. COMPROMETIDO

Existe una dependencia metodológica que puede haber alterado el resultado. Debe repetirse antes de utilizarlo como evidencia.

### 2.5. INVALIDADO COMO RESULTADO FINAL

No puede presentarse como conclusión definitiva, aunque pueda conservarse como registro histórico.

### 2.6. REQUIERE REGENERACIÓN

El artefacto depende de una etapa upstream que será corregida. Debe reconstruirse antes de continuar.

---

# 3. Resumen ejecutivo de problemas críticos

Los problemas principales son:

1. uso del calendario NASDAQ para MNQ;
2. eliminación rígida de jornadas con menos de 691 barras;
3. clasificación incorrecta de regímenes;
4. barra de las 16:00 asignada a Overnight;
5. posible reutilización silenciosa de Parquet antiguos;
6. zona horaria inferida pero no confirmada documentalmente;
7. falta de auditoría formal de rollover;
8. thresholds y targets condicionados por regímenes incorrectos;
9. dirección retrospectiva en BAR y OPC;
10. barreras evaluadas solo con cierres;
11. `NO_TRADE` heterogéneo;
12. codificación OPC inconsistente entre S04 y S07;
13. selección de features usando todo 2020–2024;
14. evaluación walk-forward posterior a una selección que ya observó 2022–2024;
15. uso extensivo de 2025 para selección;
16. probabilidades no calibradas;
17. class weights con sobrepredicción de clases minoritarias;
18. Stage 07 creado sobre metadata y mappings no verificados;
19. ausencia de backtest causal definitivo;
20. imposibilidad de resolver secuencia intrabar con OHLCV de un minuto.

Los puntos 1–18 requieren revisión antes de entrenar modelos definitivos.

---

# 4. Problemas por stage

## 4.1. S00 — Raw Data Preparation

### Problema S00-01 — Documentación no coincide con la implementación

El Markdown histórico afirmaba operaciones que S00 no ejecutaba:

- conversión de zona horaria;
- filtrado de sesión;
- eliminación de feriados;
- eliminación de días incompletos;
- continuidad minuto a minuto;
- creación del dataset intradía final.

La implementación real solo:

- lee;
- concatena;
- ordena;
- agrega contrato;
- guarda el dataset crudo.

**Estado:**

```text
documentación histórica: INVALIDADA
dataset consolidado: VÁLIDO COMO ANTECEDENTE
```

**Acción:**

Reescribir S00 para describir únicamente la ingestión cruda.

**Actualización (S00 v2 — APROBADO):**

```text
RESUELTO. notebooks/S00_raw_data_preparation_v2.ipynb es una notebook delgada
que delega toda la lógica a src/data/s00_raw_ingestion.py; no describe
operaciones que no ejecuta. La notebook v1 se conserva sin modificar, solo
como evidencia histórica.
```

---

### Problema S00-02 — Índice tz-naive

El índice de `mnq_raw` quedó sin zona horaria:

```text
df_mnq_raw.index.tz == None
```

La interpretación como UTC fue asumida posteriormente.

**Estado:**

```text
interpretación UTC: PENDIENTE DE CONFIRMACIÓN
artefacto crudo: UTILIZABLE CON AUDITORÍA
```

**Acción:**

Confirmar la zona horaria mediante configuración de NinjaTrader, exportación o proveedor.

**Actualización (S00 v2 — APROBADO):**

```text
NO RESUELTO, PERO CORREGIDO DOCUMENTALMENTE. El índice de mnq_raw_v2.parquet
sigue siendo tz-naive (no se aplicó tz_localize ni tz_convert, por decisión
de alcance de S00). config/data_config.yaml, mnq_raw_v2_manifest.json y
mnq_raw_v2_summary.json declaran explícitamente:
  timezone_stored: null
  timezone_assumption: "UTC"
  timezone_evidence: "inferred_not_confirmed"
A diferencia del summary de S00 v1 (que localizaba el índice a UTC solo para
el reporte y podía sugerir tz-aware), el summary de S00 v2 no afirma UTC
como confirmado. La confirmación documental contra la fuente/proveedor
sigue pendiente y no se resuelve con los datos disponibles.
```

---

### Problema S00-03 — Nomenclatura de contratos inconsistente

Se utilizaron dos formatos:

```text
MNQH20
H20
```

**Estado:**

```text
problema documental y de portabilidad
```

**Acción:**

Elegir una convención única y conservar metadata de contrato completa.

**Actualización (S00 v2 — APROBADO):**

```text
RESUELTO. Una única función unificada extrae instrument/contract/contract_full
desde el nombre de archivo (sin duplicación). El dataset persistido conserva
el formato corto vigente (H20, M20, U20, Z20) para no romper el contrato de
entrada actual de S01. instrument ("MNQ") y contract_full ("MNQH20") quedan
disponibles como metadata separada en mnq_raw_v2_manifest.json y
manifests/s00_source_manifest.csv, no en el dataset fila a fila.
```

---

### Problema S00-04 — Chequeo diario incorrecto

La notebook marcaba como sospechosos días con más de 500 registros, aunque el dataset crudo cubre casi 24 horas.

**Estado:**

```text
diagnóstico INVALIDADO
```

La ausencia de timestamps duplicados sigue siendo un control válido.

**Actualización (S00 v2 — APROBADO):**

```text
RESUELTO. El chequeo de ">500 filas por día" no existe en
src/data/s00_raw_ingestion.py. S00 v2 valida duplicados exactos de fila y
duplicados por (timestamp, contract) mediante comparación explícita, no
mediante un umbral de conteo diario.
```

---

### Problema S00-05 — Gap previo a MNQM25

Se identificó una discontinuidad temporal entre el final del contrato anterior y el inicio de `MNQM25`.

**Estado:**

```text
problema conocido, no resuelto
```

**Acción:**

Determinar si se trata de:

- ausencia real de archivos;
- cambio de proveedor;
- error de exportación;
- periodo sin datos;
- cambio de contrato mal documentado.

**Actualización (S00 v2 — APROBADO):**

```text
SIGUE NO RESUELTO. Cuantificado con precisión: 15d19h12min entre
2025-03-21 13:30:00 (fin de H25) y 2025-04-06 08:42:00 (inicio de M25),
15 jornadas calendario completas sin datos. Clasificado formalmente en
data/01_raw/mnq_raw_v2_gaps.parquet como:
  gap_type_structural: inter_contract
  evidence_level: unconfirmed
Interpretación provisional bajo hipótesis UTC: "sin patrón estructural
reconocido en S00; requiere auditoría adicional". No se rellenó, interpoló
ni eliminó ningún dato. La causa (archivo faltante, cambio de proveedor,
error de exportación, periodo real sin datos) sigue sin determinarse.
```

---

### Problema S00-06 — Gap interno en MNQM23 (hallazgo nuevo de S00 v2)

La auditoría que precedió a S00 v2 encontró un segundo gap extraordinario,
no documentado previamente en ningún archivo del proyecto: un salto de
**260h15min (~10d20h15min)** dentro de un único archivo/contrato —
`13_mnq_06_23.Last.txt` (M23) — entre 2023-04-05 18:03:00 y
2023-04-16 14:18:00, con 10 jornadas calendario completas sin datos. A
diferencia de S00-05, no es una transición entre archivos (`intra_file`, no
`inter_contract`).

**Estado:**

```text
hallazgo nuevo, no resuelto
```

Clasificado en `data/01_raw/mnq_raw_v2_gaps.parquet` como:

```text
gap_type_structural: intra_file
evidence_level: unconfirmed
```

**Acción:**

Misma naturaleza que S00-05: determinar si se trata de ausencia real de
datos en la fuente, error de exportación, o un evento legítimo del
proveedor. No bloquea la aprobación de S00 v2 (se documenta y se avanza),
pero debe auditarse antes de construir features/targets que crucen ese
intervalo.

---

## 4.1-bis. S00 v2 — Estado tras la reconstrucción (APROBADO)

**S00 v2 fue aprobado formalmente.** Reemplaza funcionalmente la
consolidación cruda de S00 v1 como fuente para S01 en adelante. La notebook
histórica `S00_raw_data_preparation.ipynb` (v1) **no fue modificada** y se
conserva únicamente como evidencia histórica; la implementación vigente es
`src/data/s00_raw_ingestion.py` + `notebooks/S00_raw_data_preparation_v2.ipynb`.

```text
Artefacto principal:      data/01_raw/mnq_raw_v2.parquet
Filas:                     2.172.640
Filas rechazadas:          0
Columnas:                  open, high, low, close, volume, contract
Índice:                    DatetimeIndex, tz-naive
Pruebas:                   35/35 aprobadas (25 unitarias + 10 de integración)
```

**Confirmado / persistido correctamente:**

```text
timestamps tz-naive (sin conversión de zona horaria en S00, por decisión de alcance)
timezone_assumption = UTC, declarado explícitamente como inferido, NO confirmado
timestamp_semantics (inicio vs. cierre de barra) = explícitamente no confirmado
price_type = "Last", inferido solo del nombre de archivo, sin confirmación del proveedor
```

**Pendiente, no bloqueante para la aprobación:**

```text
Gap M23 (interno, S00-06): no_resuelto
Gap H25→M25 (transición, S00-05): no_resuelto
Confirmación documental de zona horaria de origen: sigue pendiente
Chequeo automatizado de solapamiento de intervalos entre archivos: mejora
  menor pendiente (la auditoría previa lo verificó manualmente — 0
  solapamientos — pero esa verificación no quedó automatizada dentro de
  s00_raw_ingestion.py)
```

**Resuelto respecto a v1:** S00-01 (documentación desalineada), S00-03
(nomenclatura de contratos duplicada/inconsistente), S00-04 (diagnóstico de
">500 filas/día" inválido). Ver actualizaciones en cada problema arriba.

Detalle completo en `reports/stage_reports/S00_v2_report.md`.

---

## 4.2. S01 — Intraday Data Preparation

### Problema S01-01 — Calendario incorrecto

S01 utilizó:

```python
mcal.get_calendar("NASDAQ")
```

MNQ es un futuro de CME Globex.

**Impacto potencial:**

- feriados;
- cierres anticipados;
- jornadas especiales;
- fechas excluidas;
- cobertura anual;
- evaluación temporal.

**Estado:**

```text
filtrado por calendario: COMPROMETIDO
dataset intradía histórico: REQUIERE AUDITORÍA
```

No se afirma que todas las fechas sean incorrectas, pero el procedimiento no es metodológicamente aceptable como definitivo.

---

### Problema S01-02 — Filtrado temporal antes de conversión completa

Las fechas de mercado se filtraron antes de completar la localización UTC y conversión a New York.

**Estado:**

```text
procedimiento COMPROMETIDO
```

**Acción correcta:**

```text
localizar origen
→ convertir a America/New_York
→ asignar fecha operativa
→ aplicar calendario
```

---

### Problema S01-03 — Eliminación automática de jornadas incompletas

Se eliminaron 80 jornadas por no contener exactamente 691 barras o por presentar gaps.

Algunas podrían ser:

- cierres anticipados legítimos;
- sesiones especiales;
- días válidos con horario reducido.

**Estado:**

```text
política de exclusión INVALIDADA COMO REGLA DEFINITIVA
```

**Impacto:**

- sesgo de selección de sesiones;
- pérdida de eventos especiales;
- alteración de distribuciones;
- reducción artificial de escenarios extremos.

---

### Problema S01-04 — Regímenes incorrectos

La implementación histórica fue:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–15:29
4 Closing: 15:30–15:59
16:00 → valor por defecto 0
```

La convención vigente es:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–14:59
4 Closing: 15:00–16:00
```

**Estado:**

```text
regime_id histórico: INVALIDADO
```

**Impacto directo:**

- análisis por régimen;
- thresholds por régimen;
- targets;
- features de contexto;
- modelos `regime_3`;
- evaluación Closing;
- uso de `regime_id` como predictor.

---

### Problema S01-05 — Barra de las 16:00 asignada a Overnight

La barra no coincidía con ninguna condición y recibió el valor por defecto.

**Estado:**

```text
clasificación INVALIDADA
```

Todas las estadísticas de Overnight incluyen contaminación de la barra de cierre.

---

### Problema S01-06 — Reutilización silenciosa de Parquet

El código podía cargar un Parquet existente en lugar de sobrescribirlo después de modificar la notebook.

**Riesgo:**

El código visible y el artefacto real podían pertenecer a versiones diferentes.

**Estado:**

```text
reproducibilidad COMPROMETIDA
```

**Acción:**

Utilizar versionado, checksum, `FORCE_REBUILD` y metadata de creación.

---

### Problema S01-07 — Validaciones declaradas pero no implementadas

La documentación mencionaba controles OHLCV que no aparecían ejecutados.

**Estado:**

```text
afirmaciones de validación INVALIDADAS
```

Los controles deben implementarse explícitamente en la reconstrucción.

---

## 4.3. S02 — Intraday Data Analysis

### Problema S02-01 — Resultados por régimen contaminados

S02 heredó el `regime_id` incorrecto.

**Estado:**

```text
estadísticas globales: VÁLIDAS COMO ANTECEDENTE
estadísticas por año: VÁLIDAS COMO ANTECEDENTE
estadísticas por régimen: REQUIEREN REGENERACIÓN
```

Especialmente afectados:

- Regular;
- Closing;
- Overnight;
- comparaciones de estabilidad entre regímenes;
- resultados que justificaron `regime_3`.

---

### Problema S02-02 — Asimetría de ventanas

Las ventanas históricas y futuras tienen semánticas diferentes:

```text
histórica h: t-h+1 ... t
futura h: t+1 ... t+h
```

Esto es válido si se documenta, pero puede inducir interpretaciones erróneas.

**Estado:**

```text
no invalida el cálculo
requiere documentación explícita
```

---

### Problema S02-03 — “Cambios estructurales” no formales

La sección utilizó comparaciones, rolling y zonas extremas, no tests econométricos formales de ruptura.

**Estado:**

```text
diagnóstico EXPLORATORIO
```

No debe presentarse como prueba estadística de quiebre estructural.

---

### Problema S02-04 — Features exploratorias reemplazadas

El script de features de S02 fue superado por Stage 05.

**Estado:**

```text
VÁLIDO COMO ANTECEDENTE
NO ES PIPELINE OFICIAL
```

---

### Problema S02-05 — Semántica inconsistente de `body_pts`

En distintos bloques se utilizó:

```text
abs(close - open)
```

y:

```text
close - open
```

**Estado:**

```text
definición histórica AMBIGUA
```

Debe unificarse antes de reutilizar la feature.

---

## 4.4. S03 — Threshold Calibration

### Problema S03-01 — Thresholds por régimen construidos con regímenes incorrectos

Los thresholds principales dependen de `regime_id`.

**Estado:**

```text
thresholds por régimen: REQUIEREN REGENERACIÓN
thresholds globales: VÁLIDOS COMO BENCHMARK HISTÓRICO
```

Los 39 thresholds por régimen no deben utilizarse como configuración definitiva.

---

### Problema S03-02 — Closing sin H60/H90

La definición histórica de Closing contenía solo 30 minutos.

Esto produjo:

- muy pocas observaciones válidas para H30;
- ninguna para H60/H90;
- solo 39 combinaciones, no 45.

**Estado:**

```text
resultado INVALIDADO POR DEFINICIÓN DE RÉGIMEN
```

---

### Problema S03-03 — Excursiones calculadas solo con `close`

Los thresholds se calibraron con máximos y mínimos de cierres futuros, no con `high` y `low`.

**Estado:**

```text
VÁLIDO PARA UN TARGET BASADO EN CIERRES
NO VÁLIDO AUTOMÁTICAMENTE PARA EJECUCIÓN INTRABAR
```

No debe mezclarse con barreras evaluadas mediante high/low sin recalibración.

---

### Problema S03-04 — Excursiones direccionales negativas

`up_excursion` y `down_excursion` no se truncaron en cero.

**Estado:**

```text
impacto limitado sobre threshold_common
requiere revisión para thresholds direccionales
```

---

### Problema S03-05 — Reserva de 2025–2026 posteriormente observada

S03 no utilizó 2025–2026 para calibrar thresholds, pero stages posteriores sí analizaron estos años.

**Estado:**

```text
thresholds históricamente congelados: VÁLIDOS COMO ANTECEDENTE
holdout final 2025–2026: INVALIDADO
```

---

## 4.5. S04 — Operational Target Investigation

### Problema S04-01 — Targets heredaron thresholds por régimen incorrectos

DIR, BAR y OPC dependen de thresholds calculados con regímenes históricos erróneos.

**Estado:**

```text
targets históricos: REQUIEREN REGENERACIÓN
```

Esto no implica que la lógica del código sea inútil, sino que las etiquetas no deben considerarse definitivas.

---

### Problema S04-02 — Dirección retrospectiva

DIR selecciona LONG o SHORT utilizando toda la ventana futura.

BAR usa esa dirección para evaluar barreras.

OPC combina DIR y BAR.

Flujo:

```text
futuro
→ selecciona dirección
→ evalúa barrera
→ construye clase operativa
```

**Estado:**

```text
BAR como señal operativa: INVALIDADO
OPC como target operativo definitivo: COMPROMETIDO
```

Puede conservarse como formulación de investigación, pero no como representación directa de una decisión ex ante.

---

### Problema S04-03 — Alta tasa de TP inducida por la etiqueta

La tasa histórica de TP/resueltos cercana al 90 % está influida por la selección retrospectiva de dirección.

**Estado:**

```text
no puede interpretarse como win rate
```

Cualquier conclusión de rentabilidad basada en esa tasa queda invalidada.

---

### Problema S04-04 — Barreras evaluadas con cierres

TP y SL se comprobaron mediante `close`, no mediante `high` y `low`.

**Consecuencia:**

La etiqueta representa:

```text
cierre futuro alcanza nivel
```

No representa necesariamente:

```text
precio intraminuto toca nivel
```

**Estado:**

```text
VÁLIDO COMO TARGET DE CIERRES
NO VÁLIDO COMO SIMULACIÓN DE EJECUCIÓN REAL
```

---

### Problema S04-05 — `NO_TRADE` heterogéneo

`NO_TRADE` agrupa:

- ausencia de movimiento direccional suficiente;
- dirección existente sin TP ni SL.

**Estado:**

```text
target COMPROMETIDO POR HETEROGENEIDAD
```

Puede dificultar el aprendizaje y la interpretación de probabilidades.

---

### Problema S04-06 — Desbalance severo

Las clases perdedoras individuales son muy poco frecuentes.

**Estado:**

```text
problema estructural del target
```

No invalida las etiquetas, pero afecta:

- entrenamiento;
- macro-F1;
- calibración;
- estabilidad;
- sensibilidad a class weights.

---

### Problema S04-07 — Mapping OPC inconsistente con Stage 07

Mapping real de S04:

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

Mapping provisional escrito en S07:

```text
0 = LONG_TP
1 = LONG_SL
2 = SHORT_TP
3 = SHORT_SL
4 = NO_TRADE
```

**Estado:**

```text
mapping provisional S07: INVALIDADO
```

Debe leerse desde metadata real.

---

### Problema S04-08 — Preselección histórica reemplazada

S04 eligió preliminarmente:

```text
opc_p50_h30_tp15_sl10
```

Posteriormente Stage 07 priorizó:

```text
opc_p50_h60_tp15_sl10
```

**Estado:**

```text
decisión S04: HISTÓRICA, NO VIGENTE
```

---

### Problema S04-09 — Análisis del supuesto holdout

Aunque 2025–2026 no recalibró thresholds, sus distribuciones y resultados participaron en análisis descriptivos y selección.

**Estado:**

```text
holdout completamente intacto: INVALIDADO
```

---

## 4.6. S05 — Feature Engineering and Predictive Datasets

### Problema S05-01 — Selección de features antes del walk-forward

S05 utilizó todo 2020–2024 para:

- IC;
- Mutual Information;
- análisis por régimen;
- análisis por año;
- selección de targets;
- selección de feature sets.

Después, S06 evaluó folds cuyo validation era 2022, 2023 y 2024.

**Estado:**

```text
métricas walk-forward posteriores: COMPROMETIDAS
```

Los años de evaluación influyeron indirectamente en la selección previa.

---

### Problema S05-02 — Mutual Information sobre muestra aleatoria

MI se calculó sobre muestras aleatorias de 50.000 filas.

**Riesgos:**

- dependencia serial;
- composición temporal;
- inestabilidad;
- sobreponderación de determinados periodos;
- interpretación excesiva.

**Estado:**

```text
ranking MI: EXPLORATORIO
```

---

### Problema S05-03 — Niveles absolutos dominantes

`rolling_high` y `rolling_low` dominaron MI.

Pueden codificar:

- año;
- contrato;
- tendencia secular;
- régimen histórico;
- nivel nominal del índice.

**Estado:**

```text
señal univariada: SOSPECHOSA
features de nivel: REQUIEREN ABLACIÓN
```

---

### Problema S05-04 — Regime_3 incorrecto

Todos los datasets `regime_3` se construyeron con:

```text
10:30–15:29
```

La convención vigente termina a las 14:59.

**Estado:**

```text
datasets regime_3: REQUIEREN REGENERACIÓN
```

---

### Problema S05-05 — Feature sets no incluyen todo el contexto construido

Se construyeron variables temporales que no quedaron en los sets finales.

**Estado:**

```text
decisión histórica, no necesariamente error
```

Debe reevaluarse dentro del protocolo nuevo.

---

### Problema S05-06 — Metadata peligrosa dentro de Parquet

Los archivos incluyen:

- `year`;
- `dataset_split`;
- `contract`;
- `regime_id`;
- `minute_of_day`.

Una selección automática de todas las columnas podría introducir leakage o proxies temporales.

**Estado:**

```text
riesgo de implementación
```

`dataset_split` y `year` no deben entrar automáticamente al modelo.

---

### Problema S05-07 — Error temporal de detección de targets

Una regla `startswith("bar_")` clasificó features como targets en una sección intermedia.

Luego fue corregida.

**Estado:**

```text
reportes intermedios afectados
features finales no necesariamente afectadas
```

---

### Problema S05-08 — Holdout inspeccionado

Aunque IC/MI se limitaron a 2020–2024, se inspeccionaron distribuciones de 2025–2026.

**Estado:**

```text
pureza conceptual del holdout: REDUCIDA
```

---

## 4.7. S06 — Predictive Signal Analysis

### Problema S06-01 — Walk-forward no totalmente independiente

Los folds respetaron el tiempo, pero utilizaron feature sets seleccionados previamente con datos de los años evaluados.

**Estado:**

```text
resultados WF_01–WF_03: EXPLORATORIOS / OPTIMISTAS
```

No deben presentarse como estimaciones no sesgadas de generalización.

---

### Problema S06-02 — Uso extensivo de 2025

2025 se utilizó para:

- comparar candidatos;
- seleccionar configuraciones;
- analizar clases;
- estudiar meses;
- evaluar confianza;
- congelar pipelines históricos.

**Estado:**

```text
2025 como holdout ciego: INVALIDADO
```

Puede conservarse como evaluación OOS de desarrollo.

---

### Problema S06-03 — Resultados de regime_3 basados en régimen incorrecto

Los modelos y datasets específicos del régimen 3 heredaron la definición antigua.

**Estado:**

```text
resultados regime_3: REQUIEREN REGENERACIÓN
```

---

### Problema S06-04 — Logistic Regression balanceada sobrepredice pérdidas

Para BAR se observó una fuerte sobrepredicción de `SL_FIRST`.

Para OPC se sobrepredijeron clases `LONG_SL` y `SHORT_SL`.

**Estado:**

```text
class_weight="balanced": NO APROBADO COMO CONFIGURACIÓN OBLIGATORIA
```

Debe tratarse como variante experimental.

---

### Problema S06-05 — Log Loss peor que baseline por priors

Los modelos superaron algunos baselines en Balanced Accuracy y Macro-F1, pero no al Dummy basado en priors en Log Loss.

**Estado:**

```text
probabilidades históricas: NO CALIBRADAS
```

No deben utilizarse para position sizing o filtros de confianza.

---

### Problema S06-06 — Confianza BAR inversamente relacionada con acierto

En BAR se observó correlación negativa entre confianza y acierto.

**Estado:**

```text
confianza BAR como filtro: INVALIDADA
```

---

### Problema S06-07 — Confianza OPC no suficiente

Aunque la correlación fue positiva, la alta confianza se concentró en pocos casos y probablemente en `NO_TRADE`.

**Estado:**

```text
confianza OPC como probabilidad real: INVALIDADA
```

Requiere calibración OOF.

---

### Problema S06-08 — BAR más predecible, pero no operable de forma independiente

BAR mostró mejores métricas, pero su dirección proviene de información futura.

**Estado:**

```text
BAR como benchmark estadístico: VÁLIDO
BAR como señal desplegable: INVALIDADO
```

---

### Problema S06-09 — HGB rechazado con protocolo limitado

HistGradientBoosting mostró sobreajuste en la configuración probada.

**Estado:**

```text
resultado válido solo para esa configuración
```

No invalida boosting como familia de modelos.

---

## 4.8. S07_00 — Experimental Design

### Problema S07-01 — Mapping de clases incorrecto

La configuración contenía un mapping provisional distinto del real.

**Estado:**

```text
configuración de mapping: INVALIDADA
```

No debe construirse ninguna secuencia o modelo hasta corregirlo.

---

### Problema S07-02 — Feature columns no verificadas

La configuración dejó:

```text
feature_columns = null
feature_columns_verified = false
```

**Estado:**

```text
configuración incompleta
```

---

### Problema S07-03 — Nombres de columnas no reconciliados

Se mencionaron:

```text
trading_date
quarter
```

mientras los datasets históricos documentan:

```text
date
year
dataset_split
```

**Estado:**

```text
esquema físico no validado
```

---

### Problema S07-04 — Test final incorrectamente declarado

S07_00 declaró 2025–2026 como test final, aunque 2025 ya había sido utilizado.

**Estado:**

```text
declaración de holdout: INVALIDADA
```

---

### Problema S07-05 — Class weights activados obligatoriamente

La configuración fijó:

```text
use_class_weights = true
```

pese a los problemas detectados en S06.

**Estado:**

```text
decisión NO APROBADA
```

Debe convertirse en una variante experimental.

---

### Problema S07-06 — Early stopping y pérdida inconsistentes

El documento alternaba entre:

```text
categorical_crossentropy
sparse_categorical_crossentropy
```

y entre:

```text
validation_macro_f1
val_macro_f1
```

**Estado:**

```text
configuración documental INCONSISTENTE
```

---

### Problema S07-07 — Detección de raíz potencialmente infinita

La búsqueda de `PROJECT_ROOT` podía no detenerse al alcanzar la raíz del sistema.

**Estado:**

```text
bug de implementación
```

---

### Problema S07-08 — Validación solo interna de la configuración

Los archivos JSON/CSV se validaron estructuralmente, pero no contra el dataset real.

No se comprobaron:

- mapping;
- columnas;
- features;
- categorías;
- años;
- cantidades;
- continuidad;
- regímenes.

**Estado:**

```text
S07_00: CONFIGURACIÓN BASE
NO CONFIGURACIÓN EXPERIMENTAL DEFINITIVA
```

---

# 5. Problemas transversales

## 5.1. Ausencia de auditoría formal de rollover

Los contratos se consolidaron, pero no existe una auditoría definitiva de todas las transiciones.

**Impacto posible:**

- retornos artificiales;
- volatilidad;
- rolling high/low;
- secuencias;
- thresholds;
- targets;
- P&L.

**Estado:**

```text
todo análisis sensible a niveles o diferencias entre contratos: COMPROMETIDO HASTA AUDITORÍA
```

---

## 5.2. No estacionariedad y cambios de distribución

Los resultados muestran cambios importantes entre años y meses.

Especialmente:

- 2020;
- 2022;
- 2025;
- 2026 parcial.

**Estado:**

```text
problema estructural del dominio
```

No invalida el proyecto, pero exige:

- evaluación temporal;
- ventanas móviles;
- estabilidad por contexto;
- recalibración controlada;
- cautela con niveles absolutos.

---

## 5.3. Dependencia serial y labels solapados

Observaciones de un minuto y targets H30/H60/H90 se solapan ampliamente.

**Estado:**

```text
muestras no independientes
```

Consecuencias:

- métricas con varianza subestimada;
- breadth exagerado;
- bootstrap/OOB inadecuado;
- significancia univariada inflada;
- necesidad de revisar purging en cortes internos.

---

## 5.4. Falta de test final verdaderamente intacto

2025 fue utilizado extensamente.

2026 es parcial.

**Estado:**

```text
no existe un holdout histórico amplio completamente intacto
```

La reconstrucción debe:

- congelar decisiones antes de 2026;
- declarar 2026 como test parcial;
- considerar validación prospectiva futura.

---

## 5.5. Múltiples pruebas no registradas completamente

Se probaron múltiples:

- targets;
- thresholds;
- horizons;
- feature sets;
- scopes;
- modelos;
- pesos;
- análisis.

No existe todavía un registro completo del número efectivo de pruebas.

**Estado:**

```text
riesgo de data snooping
```

---

## 5.6. Ausencia de backtest causal definitivo

No existe aún una simulación definitiva que integre:

- momento real de disponibilidad de la señal;
- entrada posterior;
- spread;
- slippage;
- comisiones;
- tasas;
- rollover;
- ambigüedad intrabar;
- órdenes pendientes;
- restricciones de posición.

**Estado:**

```text
ningún resultado predictivo demuestra rentabilidad
```

---

## 5.7. Limitaciones irreducibles de OHLCV de un minuto

No puede conocerse:

- orden exacto de high y low;
- bid/ask;
- spread real;
- profundidad;
- fills;
- secuencia de trades;
- dirección agresora;
- VWAP real.

**Estado:**

```text
limitación estructural, no corregible con código
```

Debe manejarse mediante reglas conservadoras o adquisición de datos más granulares.

---

# 6. Resultados específicamente invalidados

Los siguientes enunciados no deben utilizarse:

```text
“Los regímenes históricos están correctamente definidos.”
“Closing tiene solo 30 minutos.”
“La barra de las 16:00 pertenece a Overnight.”
“El calendario NASDAQ es apropiado para MNQ.”
“Todas las jornadas con menos de 691 barras son inválidas.”
“2025–2026 constituye un holdout ciego.”
“BAR presenta un win rate cercano al 90 %.”
“OPC representa directamente una decisión ex ante.”
“Las probabilidades históricas están calibradas.”
“La confianza del modelo puede utilizarse como filtro operativo.”
“class_weight=balanced debe usarse obligatoriamente.”
“El mapping OPC de S07 es correcto.”
“Los folds de S06 son completamente independientes.”
“Las features seleccionadas en S05 fueron elegidas sin observar los años de validación.”
“Los resultados de regime_3 representan el régimen Regular vigente.”
“S07_00 está listo para entrenar modelos definitivos.”
“Un mejor Macro-F1 demuestra rentabilidad.”
```

---

# 7. Resultados que pueden conservarse como antecedentes

Pueden conservarse, con sus advertencias:

### S00

- consolidación de 26 archivos;
- ausencia de timestamps duplicados;
- ausencia de superposición entre archivos;
- inventario de contratos;
- tamaño y cobertura del dataset crudo.

### S01

- ventana objetivo 04:30–16:00;
- conversión conceptual a America/New_York;
- necesidad de continuidad minuto a minuto;
- estructura de columnas del dataset intradía.

### S02

- análisis global de horizontes 30/60/90;
- cantidad de muestras válidas;
- evidencia exploratoria a favor de h60;
- necesidad de walk-forward;
- no cruzar jornadas;
- existencia de cambios de distribución.

### S03

- metodología de excursiones futuras;
- comparación de p40/p50/p60;
- benchmark global;
- idea de thresholds condicionados por contexto;
- necesidad de congelar thresholds dentro del train.

### S04

- código y estructura conceptual de DIR/BAR/OPC;
- mapping real de clases;
- evidencia de desbalance;
- advertencia sobre `NO_TRADE`;
- necesidad de separar etiqueta y ejecución.

### S05

- biblioteca de 116 features causales;
- identificación de redundancia;
- comparación con/sin niveles;
- importancia de rangos y volatilidad;
- necesidad de seleccionar dentro del train.

### S06

- Logistic Regression como baseline histórico;
- evidencia de que BAR es más predecible que OPC bajo la etiqueta histórica;
- evidencia de inestabilidad mensual;
- problemas de calibración;
- utilidad de reportar por clase y fold.

### S07

- arquitectura modular del Stage 07;
- necesidad de dataset común para todos los modelos;
- validación interna temporal;
- separación entre entrenamiento y evaluación externa;
- catálogo de modelos;
- estructura de artefactos.

---

# 8. Artefactos que deben regenerarse

Si se corrigen calendario, jornadas o regímenes, deben regenerarse:

```text
S01:
mnq_intraday.parquet
mnq_intraday_summary.json
```

Luego, por dependencia:

```text
S02:
análisis por régimen
ventanas y resúmenes dependientes del dataset
```

```text
S03:
excursiones
thresholds globales si cambia la muestra
thresholds por régimen
candidatos finales
```

```text
S04:
DIR
BAR
OPC
metadata
resúmenes
mapeos verificados
```

```text
S05:
features
selección dentro de cada fold
datasets predictivos
distribuciones de clases
```

```text
S06:
Mutual Information por fold
baselines
métricas walk-forward
comparaciones de scopes
calibración
```

```text
S07:
configuración
secuencias
modelos
predicciones
métricas
reportes
```

No deben mezclarse artefactos nuevos upstream con resultados históricos downstream.

---

# 9. Matriz de propagación

| Problema upstream | S02 | S03 | S04 | S05 | S06 | S07 |
|---|---:|---:|---:|---:|---:|---:|
| Calendario incorrecto | Sí | Sí | Sí | Sí | Sí | Sí |
| Jornadas eliminadas | Sí | Sí | Sí | Sí | Sí | Sí |
| Regímenes incorrectos | Sí | Sí | Sí | Sí | Sí | Sí |
| Barra 16:00 mal asignada | Sí | Sí | Sí | Sí | Sí | Sí |
| Rollover no auditado | Sí | Sí | Sí | Sí | Sí | Sí |
| Targets retrospectivos | No | No | Sí | Sí | Sí | Sí |
| Selección global de features | No | No | No | Sí | Sí | Sí |
| Uso de 2025 | No | No | Parcial | Parcial | Sí | Sí |
| Mapping OPC incorrecto | No | No | Fuente real | No | No | Sí |

---

# 10. Reglas de uso para Claude

Claude debe:

1. tratar los archivos S00–S07 como documentación histórica;
2. consultar este archivo antes de reutilizar cualquier resultado;
3. no combinar artefactos de versiones diferentes;
4. identificar dependencias upstream;
5. marcar como exploratoria toda métrica histórica comprometida;
6. no afirmar que un resultado fue invalidado si solo requiere repetición;
7. conservar resultados negativos;
8. diferenciar bug de implementación, limitación de datos y decisión metodológica;
9. solicitar o verificar metadata antes de asumir mappings o columnas;
10. regenerar downstream después de cambios estructurales upstream.

Claude no debe:

- continuar directamente desde S07_01;
- asumir que el dataset histórico es definitivo;
- usar 2025 como test ciego;
- usar probabilidades históricas para sizing;
- interpretar BAR como señal de trading;
- inferir ejecución intrabar desde OHLCV;
- seleccionar modelos por P&L externo;
- ocultar que las métricas históricas están condicionadas.

---

# 11. Prioridad de corrección

## Prioridad 1 — Datos

1. confirmar timezone;
2. definir calendario CME;
3. auditar jornadas especiales;
4. corregir regímenes;
5. auditar contratos y rollover;
6. reconstruir `mnq_intraday`.

## Prioridad 2 — Labels

1. recalcular thresholds;
2. revisar close frente a high/low;
3. revisar DIR retrospectivo;
4. revisar BAR;
5. rediseñar OPC o alternativas;
6. verificar mapping.

## Prioridad 3 — Validación

1. separar train interno;
2. seleccionar features dentro del train;
3. congelar 2026;
4. definir purging según intervalos;
5. registrar todas las pruebas.

## Prioridad 4 — Modelos

1. baselines;
2. boosting;
3. MLP;
4. CNN1D;
5. GRU;
6. LSTM;
7. TCN;
8. calibración.

## Prioridad 5 — Trading

1. señal causal;
2. entrada ejecutable;
3. costes;
4. ambigüedad intrabar;
5. rollover;
6. P&L y riesgo.

---

# 12. Conclusión

El pipeline histórico contiene conocimiento valioso, pero no debe continuarse como si todos sus artefactos fueran definitivos.

La decisión correcta es:

```text
conservar documentación y resultados exploratorios
→ corregir problemas upstream
→ regenerar artefactos dependientes
→ repetir selección y evaluación temporal
→ entrenar modelos definitivos
→ realizar backtesting causal
```

Los resultados históricos deben utilizarse para orientar y auditar la reconstrucción, no para evitarla.
