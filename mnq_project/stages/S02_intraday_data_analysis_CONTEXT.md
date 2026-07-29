# S02 — Intraday Data Analysis

## 1. Identificación

- **Notebook:** `S02_intraday_data_analysis.ipynb`
- **Etapa:** Stage 02
- **Función:** análisis exploratorio integral del dataset intradiario, estudio de ventanas operativas y definición de reglas metodológicas para las etapas posteriores.
- **Estado:** ejecutada. Sus conclusiones globales siguen siendo útiles, pero los análisis por régimen deben revisarse por una inconsistencia heredada de S01.

## 2. Posición dentro del pipeline

Entrada:

```text
data/02_mnq_intraday/mnq_intraday.parquet
data/02_mnq_intraday/mnq_intraday_summary.json
```

Dependencia directa:

```text
S00 → S01 → S02
```

S02 no construye todavía los targets definitivos ni entrena modelos. Su objetivo es comprender la estructura del mercado y preparar criterios para thresholds, targets, features y validación temporal.

## 3. Dataset analizado

```text
Shape: 1.024.062 × 9
Periodo: 2020-01-02 04:30 → 2026-04-17 16:00
Zona horaria: America/New_York
Trading days: 1.482
Barras por día: 691
Frecuencia: 1 minuto
Índice: ordenado, único y tz-aware
Duplicados temporales: 0
Gaps intradiarios: 0
```

Días disponibles por año:

```text
2020: 225
2021: 244
2022: 240
2023: 237
2024: 239
2025: 227
2026: 70, periodo parcial hasta el 17 de abril
```

## 4. Análisis implementados

La notebook desarrolla:

1. validación temporal y estructural;
2. cobertura por año, trimestre, mes y contrato;
3. distribución por régimen intradiario;
4. construcción y validación de 24 métricas OHLCV;
5. estadística descriptiva, percentiles, correlaciones y extremos;
6. ventanas históricas y futuras de 30, 60 y 90 minutos;
7. análisis por año, régimen, contrato y trimestre;
8. análisis de estabilidad entre contextos;
9. análisis rolling de 20, 60 y 120 días;
10. diagnóstico de posibles cambios estructurales;
11. diseño preliminar de validación temporal;
12. generación exploratoria de variables candidatas.

## 5. Métricas OHLCV

Se construyeron 24 métricas descriptivas agrupadas en:

```text
cambios de precio y retornos
rango y estructura de vela
mechas y posición del cierre
volumen y transformaciones logarítmicas
```

Resultados de validación:

```text
Errores OHLC críticos: 0
Valores infinitos: 0
Volumen negativo: 0
Precios no positivos: 0
NaN por primera barra diaria: 1.482, esperado
Velas con rango cero: 30
```

Los 30 rangos iguales a cero generan NaN esperados en ratios de vela.

## 6. Ventanas operativas

Horizontes evaluados:

```text
30, 60 y 90 minutos
```

Reglas:

- las ventanas se calculan dentro del mismo `date`;
- la ventana histórica incluye la barra actual;
- la ventana futura comienza en `t+1`;
- ninguna ventana cruza días;
- las métricas futuras no pueden usarse como features.

### Validez de ventanas

| Horizonte | Históricas válidas | Futuras válidas | Histórica + futura |
|---:|---:|---:|---:|
| 30 | 981.084 | 979.602 | 936.624 |
| 60 | 936.624 | 935.142 | 847.704 |
| 90 | 892.164 | 890.682 | 758.784 |

Porcentaje con ambas ventanas completas:

```text
30m: 91,46 %
60m: 82,78 %
90m: 74,10 %
```

### Definición exacta

Para una ventana histórica de tamaño `h`:

```text
t-h+1, ..., t
```

Son `h` barras, pero el retorno entre el primer y último cierre abarca `h-1` intervalos.

Para una ventana futura de tamaño `h`:

```text
t+1, ..., t+h
```

Son `h` barras y `h` intervalos respecto del cierre en `t`.

Esta asimetría es intencional en el código, pero debe recordarse al comparar métricas históricas y futuras.

## 7. Resultados principales por horizonte

Medianas globales:

| Horizonte | Retorno absoluto futuro | Excursión máxima futura | Rango futuro |
|---:|---:|---:|---:|
| 30m | 16,25 | 30,75 | 40,75 |
| 60m | 23,75 | 45,00 | 59,75 |
| 90m | 30,00 | 57,00 | 75,50 |

Percentil 95 de excursión máxima:

```text
30m: 106,25 puntos
60m: 151,50 puntos
90m: 186,50 puntos
```

Hallazgos:

- el salto relativo más importante aparece de 30 a 60 minutos;
- 90 minutos agrega movimiento, pero con menor ganancia relativa y menos muestras;
- la excursión máxima mediana es aproximadamente 1,9 veces el retorno absoluto final;
- las excursiones bajistas extremas son ligeramente mayores que las alcistas;
- 60 minutos aparece como equilibrio razonable entre magnitud y disponibilidad.

Este análisis fue uno de los antecedentes para priorizar posteriormente el horizonte `h60`.

## 8. Hallazgos temporales

La dinámica del MNQ no es homogénea.

Los análisis por año, trimestre, rolling y contexto señalan como periodos especialmente activos:

```text
2022
2025Q2
2025Q3
2026Q1
2026Q2 parcial
```

Al final del dataset, las métricas rolling de excursión aparecen en zona extrema para los horizontes y ventanas evaluados.

Interpretación vigente:

- existen periodos tranquilos, activos y extremos;
- el promedio global puede ocultar inestabilidad;
- los modelos deben evaluarse por contexto;
- 2026 debe interpretarse con cautela porque solo contiene 70 jornadas.

La sección de “cambios estructurales” constituye un diagnóstico exploratorio basado en comparaciones temporales, rolling y zonas extremas. No debe interpretarse como una prueba econométrica formal de ruptura estructural.

## 9. Decisiones metodológicas derivadas

S02 estableció principios que continúan vigentes:

```text
No utilizar splits aleatorios.
Entrenar con pasado y evaluar con futuro.
Usar validación walk-forward.
Reservar 2025–2026 para evaluación final.
No utilizar variables future_* como features.
Ajustar transformaciones únicamente con train.
Evaluar resultados globales y por contexto.
Empezar con un modelo general, pero no ciego al contexto.
```

Diseño walk-forward preliminar, posteriormente ratificado:

```text
WF_01: train 2020–2021 → validation 2022
WF_02: train 2020–2022 → validation 2023
WF_03: train 2020–2023 → validation 2024
```

Test final propuesto:

```text
Train final: 2020–2024
Test final: 2025–2026
```

## 10. Variables candidatas generadas

La notebook crea un bloque exploratorio reutilizable:

```text
mnq_candidate_feature_engineering.py
mnq_candidate_feature_groups.json
```

Resultado ejecutado:

```text
df_candidate_features: 1.024.062 × 43
```

Grupos declarados:

```text
intraday_context: 2
historical_context: 9
candle_volatility_context: 9
volume_context: 7
rolling_market_context: 6
broad_temporal_context: 5
```

Las métricas rolling diarias usan `shift(1)`, por lo que incorporan únicamente jornadas completas anteriores.

**Estado actual:** este bloque fue exploratorio. El pipeline oficial de features se desarrolló posteriormente en Stage_05 y debe considerarse la fuente vigente.

## 11. Artefactos generados

Carpeta:

```text
data/02_mnq_intraday/s02_intraday_data_analysis_results/
```

Principales archivos:

```text
df_window_metrics.parquet                   1.024.062 × 156
df_stability_summary.parquet                      72 × 27
df_stability_compact.parquet                      72 × 14
df_rolling_summary.parquet                        54 × 17
df_rolling_main_summary.parquet                   18 × 17
df_rolling_extreme_periods.parquet                90 × 10
df_regime_rolling_summary.parquet                 39 × 11
df_structural_compact.parquet                     12 × 14
df_structural_extreme_zones.parquet               64 × 13
df_structural_recent_state.parquet                 9 × 9
s02_intraday_analysis_summary.json
s02_saved_files_manifest.csv
mnq_candidate_feature_engineering.py
mnq_candidate_feature_groups.json
```

`df_window_metrics.parquet` es un artefacto pesado generado y no debe incorporarse al repositorio ni al contexto permanente de Claude.

## 12. Problemas críticos

### 12.1. Regímenes heredados de S01

La distribución observada en S02 es:

```text
Overnight: 04:30–08:29 y también 16:00
Pre-market: 08:30–09:29
Opening: 09:30–10:29
Regular: 10:30–15:29
Closing: 15:30–15:59
```

La barra de `16:00` quedó clasificada como `Overnight`, porque S01 le asignó el valor por defecto `regime_id = 0`.

La convención aprobada posteriormente es:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–14:59
4 Closing: 15:00–16:00
```

Consecuencia:

- los análisis globales y por año siguen siendo útiles;
- todas las estadísticas, rankings y conclusiones segmentadas por régimen deben recalcularse o interpretarse con cautela;
- `Regular` y `Closing` están especialmente afectados.

### 12.2. Dependencia de S01

S02 asume que `mnq_intraday` es definitivo. Por lo tanto, hereda las cuestiones pendientes de S01 sobre:

```text
calendario NASDAQ usado para MNQ
jornadas incompletas eliminadas
definición de regímenes
```

### 12.3. Semántica inconsistente de `body_pts`

En la sección OHLCV:

```text
body_pts = abs(close - open)
```

En el script exploratorio de features:

```text
body_pts = close - open
abs_body_pts = abs(body_pts)
```

Las etapas posteriores deben usar nombres y definiciones consistentes.

### 12.4. Nomenclatura histórica de targets

S02 menciona targets preliminares:

```text
T2, T4 y T5
```

Esa nomenclatura fue reemplazada posteriormente por:

```text
DIR, BAR y OPC
```

El target vigente para Stage_07 es:

```text
opc_p50_h60_tp15_sl10
```

## 13. Relación con el estado actual

Elementos de S02 que continúan vigentes:

```text
horizontes candidatos 30/60/90
prioridad analítica de 60 minutos
ventanas sin cruzar días
separación entre histórico y futuro
validación walk-forward
reserva de 2025–2026
evaluación por contexto
prohibición de leakage
```

Elementos reemplazados:

```text
T2/T4/T5 → DIR/BAR/OPC
features candidatas S02 → feature engineering oficial Stage_05
diseño exploratorio → configuración formal Stage_06 y Stage_07
regímenes originales → convención corregida pendiente de consolidación
```

## 14. Estado y acciones pendientes

**Aprobado:**

- análisis global del dataset;
- definición causal de ventanas futuras;
- horizontes 30/60/90;
- evidencia para continuar estudiando h60;
- reglas de validación temporal y leakage;
- reserva de 2025–2026.

**Pendiente:**

1. corregir los regímenes en la fuente;
2. recalcular resultados dependientes de `regime_id`;
3. decidir si se reconstruirán los artefactos posteriores;
4. unificar la definición de `body_pts`;
5. mantener los archivos pesados fuera de GitHub;
6. tratar el script de features de S02 únicamente como antecedente;
7. conservar como oficiales los targets y features definidos en stages posteriores.

Todo lo anterior en este documento describe **exclusivamente la notebook
v1** (`S02_intraday_data_analysis.ipynb`), que no fue modificada y se
conserva como evidencia histórica. Sus resultados (régimen contaminado,
cobertura asumida 691 barras/día siempre, `body_pts` ambiguo, generador
exploratorio de features, "cambios estructurales" no formales, afirmación de
holdout ciego 2025–2026) **no deben presentarse como vigentes.** El cierre
real de la etapa se realizó mediante una implementación nueva (v2) — ver
§15.

---

## 15. S02 v2 — Cierre aprobado (actualización)

**S02 v2 fue aprobado formalmente**, con estado
`APPROVED_WITH_KNOWN_LIMITATION` (limitación técnica conocida declarada
explícitamente, no bloqueante). Es la implementación vigente.

```text
Implementación:    src/data/s02_intraday_analysis.py
Config:            config/s02_analysis_config.yaml
Notebook vigente:  notebooks/S02_intraday_data_analysis_v2.ipynb
Reporte:           reports/stage_reports/S02_v2_report.md
Manifiesto:        data/02_intraday/s02_analysis_manifest.json
Pruebas:           26/26 unitarias + 12/12 integración (114/115 en la suite
                   completa del repo; única falla preexistente de S00, no
                   relacionada)
```

### Resolución de los 7 pendientes de §14

| # | Pendiente (v1) | Resolución en S02 v2 |
|---|---|---|
| 1 | Corregir los regímenes en la fuente | No corresponde a S02 (ya resuelto en S01 v2); S02 v2 consume `regime_label` vigente de S01 v2 sin recalcularlo |
| 2 | Recalcular resultados dependientes de `regime_id` | Resuelto: `s02_regime_distribution`, `s02_window_metrics_summary`, `s02_stability_summary` y ACF/Ljung-Box por régimen regenerados contra el régimen corregido (`Closing` 4,34%→8,83%) |
| 3 | Decidir si se reconstruirán los artefactos posteriores | Decidido y ejecutado: se reconstruyó S02 completo; artefactos pesados por-barra (`df_window_metrics`) reemplazados por resúmenes agregados versionables, nunca persistidos como tabla masiva |
| 4 | Unificar la definición de `body_pts` | Resuelto: `body_signed_pts` (con signo) y `body_abs_pts` (valor absoluto) en todo el código y artefactos; `body_pts` a secas no existe en S02 v2 |
| 5 | Mantener los archivos pesados fuera de GitHub | Resuelto: `data/` gitignorado (mismo patrón que S00/S01); ningún artefacto por-barra se genera ni se versiona |
| 6 | Tratar el script de features de S02 únicamente como antecedente | Resuelto: el generador exploratorio no se ejecuta ni se reproduce en S02 v2; permanece solo dentro de la notebook v1 intacta |
| 7 | Conservar como oficiales los targets y features definidos en stages posteriores | Aplicado: S02 v2 no construye, evalúa ni selecciona DIR/BAR/OPC ni ninguna feature definitiva; solo menciona a S04/S05 como responsables |

### Validación focalizada adicional (post-aprobación inicial)

Antes del cierre se verificaron tres posibles inconsistencias:

```text
1. Validez de ventanas: confirmado correcto (0 diferencia contra la formula
   teorica en las 9 filas de s02_window_validity_summary.parquet); el unico
   problema era una redaccion ambigua en un resumen previo, corregida.
2. Closing / h=90: confirmado que no hay restriccion implicita de regimen;
   causa exacta = "insufficient_bars" (el dia termina antes de completar 90
   minutos futuros), nunca "contract_change".
3. ACF/Ljung-Box: confirmado que SI formaban pares espurios entre segmentos
   distintos tras el dropna. CORREGIDO (gap-aware). Efecto medido pequeno,
   ninguna conclusion cambia.
```

### Limitación técnica conocida (no bloqueante)

```text
ARCH-LM no es gap-aware (misma clase de problema que tenia Ljung-Box antes
de corregirse). Sus resultados son unicamente diagnosticos exploratorios
sobre heterocedasticidad condicional; no deben usarse como evidencia
definitiva ni como criterio de seleccion de features o modelos. Correccion
pendiente como tarea de mantenimiento metodologico posterior.
```

### Cifras verificadas por ejecución real

```text
Filas totales (poblacion 'all'):        1.087.777
Jornadas full_day_eligible:              1.482 (691 barras/dia exactas)
Filas full_day_eligible:                 1.024.062 (identico al historico v1)
Filas partial_regime_eligible:              40.539 (119 dias)
Transiciones de contrato reales:                25
Transiciones dentro de un mismo segmento:        0
Validez de ventana full_day_eligible (full):  30m 91,46% · 60m 82,78% · 90m 74,10%
```

### Pendiente que sigue abierto (no bloqueante)

```text
Zona horaria de origen y timestamp_semantics: sin confirmacion documental
  (heredado de S00/S01).
244 jornadas partial_undetermined + 11 no_data_undetermined: causa sin
  determinar (heredado de S01).
Auditoria general de rollover (Fase 4 del rebuild plan): pendiente como
  etapa separada; S02 v2 solo aporto el impacto a nivel de ventana.
ARCH-LM no gap-aware (ver limitacion arriba).
```

Detalle completo en `reports/stage_reports/S02_v2_report.md`,
`01_CURRENT_DECISIONS.md §33` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.3-bis`.
