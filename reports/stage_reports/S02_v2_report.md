# S02 v2 — Reporte de análisis intradía MNQ

Implementación completa de S02 v2 (análisis exploratorio intradía) a partir de
los artefactos aprobados de S01 v2, más una **validación focalizada
post-aprobación inicial** (§7) que encontró y corrigió un problema real
(contaminación entre segmentos en ACF/Ljung-Box) y descartó dos falsos
positivos (validez de ventanas: solo error de redacción; `Closing`/h=90: sin
restricción implícita de régimen, causa exacta confirmada). No se tocó S00,
S01, `config/data_config.yaml`, `config/intraday_config.yaml`, la notebook
histórica de S02 ni ningún documento maestro. Sin commit ni push.

---

## 1. Alcance y decisiones vigentes aplicadas

Esta reconstrucción implementa las decisiones aceptadas por el usuario tras la
revisión técnica de la notebook histórica:

```
Poblaciones: all (cobertura/calidad/descriptivo) · full_day_eligible (principal)
             · partial_regime_eligible (secundaria, solo regímenes consecutivos)
             · descriptive_only (solo descripción) · not_model_eligible (excluida)
Ventanas: mismo date + mismo consecutive_segment_id (recalculado sobre la
          población filtrada) + mismo contract + consecutividad estricta de
          minutos. Cambio de contrato: bloqueante.
Rollover: auditoría resumida a nivel de ventana (no duplica la auditoría
          general de la Fase 4 del rebuild plan).
Features: generador exploratorio histórico NO se ejecuta ni se reproduce aquí.
Estabilidad: "inestabilidad temporal" / "cambio de distribución", nunca
             "cambio estructural"; sin pruebas formales de ruptura.
Dependencia temporal: ACF, Ljung-Box, ARCH-LM, diagnóstico acotado, global y
                       por régimen, no usado para seleccionar features.
Targets: no se construyen, evalúan ni seleccionan DIR/BAR/OPC.
```

---

## 2. Archivos creados / modificados

Creados en la primera entrega, **modificados** durante la validación
focalizada (§7): `src/data/s02_intraday_analysis.py` (ACF/Ljung-Box
gap-aware) y `tests/test_s02_intraday_analysis.py` (5 pruebas nuevas de
gap-awareness). El resto de archivos y artefactos abajo listados se
regeneraron con contenido actualizado pero sin cambio de esquema salvo
`s02_acf_summary.parquet` (columna nueva `n_pairs_effective`) y
`s02_dependence_tests_summary.parquet` (columna nueva
`n_pairs_effective_at_max_lag`).

```
config/s02_analysis_config.yaml
src/data/s02_intraday_analysis.py                                    [modificado]
tests/test_s02_intraday_analysis.py       (26 pruebas unitarias)     [modificado]
tests/test_s02_integration.py             (12 pruebas de integración)
notebooks/S02_intraday_data_analysis_v2.ipynb
data/02_intraday/s02_summary.parquet
data/02_intraday/s02_coverage_summary.parquet
data/02_intraday/s02_regime_distribution.parquet
data/02_intraday/s02_ohlcv_stats_summary.parquet
data/02_intraday/s02_ohlcv_correlation.parquet
data/02_intraday/s02_window_validity_summary.parquet
data/02_intraday/s02_rollover_window_audit.parquet
data/02_intraday/s02_window_metrics_summary.parquet
data/02_intraday/s02_stability_summary.parquet
data/02_intraday/s02_rolling_summary.parquet
data/02_intraday/s02_temporal_instability_zones.parquet
data/02_intraday/s02_acf_summary.parquet
data/02_intraday/s02_dependence_tests_summary.parquet
data/02_intraday/s02_analysis_manifest.json
reports/stage_reports/S02_v2_report.md
```

**Modificados (validación focalizada, §7):** `src/data/s02_intraday_analysis.py`,
`tests/test_s02_intraday_analysis.py`. **Historia intacta:**
`notebooks/S02_intraday_data_analysis.ipynb` (v1) y
`mnq_project/stages/S02_intraday_data_analysis_CONTEXT.md` no fueron
tocados; el generador exploratorio de features de la v1 no se ejecutó ni se
reprodujo en S02 v2.

`data/` está en `.gitignore` (mismo patrón que S00/S01), por eso los artefactos
Parquet/JSON no aparecen en `git status`.

---

## 3. Resultados comprobados por ejecución

Todo lo listado en esta sección proviene de una ejecución real (suite de
pruebas + notebook ejecutada de punta a punta desde kernel limpio, con
`nbclient`, kernel `python3` del entorno conda `stage_02`), no de inferencia.

### 3.1. Pruebas

```
tests/test_s02_intraday_analysis.py   26/26 PASSED   (21 originales + 5 nuevas de gap-awareness, §7.3)
tests/test_s02_integration.py         12/12 PASSED
Suite completa del repo (pytest -v)   114/115 PASSED
```

La única falla es `tests/test_s00_integration.py::test_never_writes_to_productive_raw_dir`,
**preexistente y no relacionada** con este trabajo (documentada en
`01_CURRENT_DECISIONS.md §31` y `02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.1-bis`
desde la aprobación de S00 v2; asume `data/01_raw/` vacío, que ya contiene
artefactos productivos). `git diff --stat` sobre cualquier archivo de S00/S01
está vacío — no se tocó nada de esas etapas.

### 3.2. Bug de rendimiento encontrado y corregido durante la implementación

`statsmodels.stats.diagnostic.acorr_ljungbox` calcula internamente la ACF sin
FFT (`O(n²)`): medido en **~30 segundos por serie** para n≈420.000
observaciones en esta máquina. Con 3 series × ~6 alcances (global + 5
regímenes) por población, el pipeline completo no terminaba en un tiempo
razonable. Se reemplazó por un cálculo manual del estadístico de Ljung-Box a
partir de la ACF vía FFT (`statsmodels.tsa.stattools.acf(..., fft=True)`,
**<0.1s** para el mismo tamaño), verificado para producir el **valor idéntico**
que `acorr_ljungbox` en muestras pequeñas antes de aplicarlo a escala completa.
Tiempo total del pipeline tras la corrección: **~128 segundos** para el
dataset completo (1.087.777 filas).

### 3.3. Validación temporal (gap-aware)

```
n_rows: 1.087.777   n_dates: 1.787   n_segments: 2.285
index monotonic / unique: True / True
date == index.date: True
segments_are_strictly_consecutive_minutes: True (0 pasos corruptos)
no_duplicate_minute_of_day_within_day: True
n_intraday_gaps_within_date_informational: 498 (esperado: jornadas parciales de v2)
VALIDACIÓN: OK, 0 checks críticos fallidos
```

### 3.4. Poblaciones

```
all                       1.087.777 filas   1.787 fechas
full_day_eligible         1.024.062 filas   1.482 días  (691.0 barras/día exactas)
partial_regime_eligible      40.539 filas     119 días  (516.6 barras/día en promedio, solo segmentos elegibles)
descriptive_only               1.359 filas     182 días
not_model_eligible (en 'all')     877 filas       4 días
```

`full_day_eligible` reproduce **exactamente** 1.482 × 691 = 1.024.062, idéntico
al histórico v1/S02 v1.

### 3.5. Régimen intradiario (corregido, `full_day_eligible`)

```
Early_Premarket   355.680 barras   34.73%
Premarket          88.920 barras    8.68%
Opening            88.920 barras    8.68%
Regular           400.140 barras   39.07%
Closing            90.402 barras    8.83%
```

Comparado con la narrativa histórica contaminada (`Regular` 43.42%, `Overnight`
34.88% incluyendo la barra 16:00, `Closing` 4.34%), `Regular` baja ~4.3 puntos
y `Closing` casi se duplica (4.34% → 8.83%) al recibir su ventana completa
correcta (15:00–16:00, 61 barras) en vez de los 30 minutos históricos.

### 3.6. Ventanas 30/60/90 — validez, `full_day_eligible`

```
Horizonte   Históricas válidas   Futuras válidas   Ambas (full)
30m         981.084 (95.80%)     979.602 (95.66%)  936.624 (91.46%)
60m         936.624 (91.46%)     935.142 (91.32%)  847.704 (82.78%)
90m         892.164 (87.12%)     890.682 (86.98%)  758.784 (74.10%)
```

Estas cifras son **idénticas** a las documentadas en
`S02_intraday_data_analysis_CONTEXT.md §6` para la notebook histórica. La
razón: en los datos reales de MNQ, **ninguna de las 25 transiciones de
contrato detectadas ocurre dentro de un mismo `consecutive_segment_id`**
(`s02_rollover_window_audit.parquet`: 0 de 25 transiciones son
`intra_segment_transition=True`) — todos los rollovers observados caen fuera
de sesión, exactamente donde ya los invalida el cambio de segmento. El nuevo
bloqueo por contrato único está implementado y probado (ver `test_window_validity_crosses_contract_change_is_blocking`
y `test_rollover_window_audit_counts_expected_bars`), pero **no reduce ninguna
cifra adicional** sobre este dataset real.

Sobre `partial_regime_eligible` (población nueva, sin antecedente v1):

```
30m: 90.34% hist · 90.01% fut · 80.35% full
60m: 80.35% hist · 80.02% fut · 64.71% full
90m: 72.48% hist · 72.22% fut · 49.46% full
```

### 3.7. Excursión máxima futura (`future_max_excursion_pts`, overall, `full_day_eligible`)

```
Horizonte   n válidas   Mediana   p95      Máximo
30m         979.602     30,75     106,25   1.463,25
60m         935.142     45,00     151,50   1.654,00
90m         890.682     57,00     186,50   1.654,00
```

Coincide exactamente con las medianas y p95 documentados para la notebook
histórica (30,75/106,25; 45,00/151,50; 57,00/186,50) — la evidencia global
sobre magnitud de movimiento por horizonte se reproduce, ahora sobre una base
metodológicamente corregida (régimen correcto, ventanas gap-aware, contrato
único verificado).

### 3.8. Estabilidad (CV) — `future_max_excursion_pts`

Los cortes más inestables (`p50_cv`) son consistentemente **régimen** y
**contrato**, no año:

```
regimen  h90  p50_cv=0.381  inestable   (top: Premarket 98,50 · low: Early_Premarket 38,75; n_groups=4, Closing sin muestra a h90)
regimen  h30  p50_cv=0.374  inestable   (top: Opening · low: Early_Premarket)
contrato h90  p50_cv=0.305  inestable   (top: M26 · low: H20)
año      h90  p50_cv=0.297  moderada    (top: 2026 · low: 2021)
```

### 3.9. Limitación estructural real de `Closing` a horizontes largos

`Closing` (15:00–16:00, 61 barras) no participa del corte por régimen a h=90
(`n_groups=4` en vez de 5 en la tabla de estabilidad) y su muestra a h=60 es
mínima (`count=1.482`, es decir 1 barra válida por día: solo el minuto 15:00
alcanza a tener 60 minutos futuros dentro de la misma jornada). Esto **ya no
es el bug histórico** (Closing mal definido con solo 30 minutos) sino una
**limitación estructural real** del régimen corregido: 61 minutos no alcanzan
para una ventana futura completa de 90 minutos sin cruzar el fin de la
jornada, y apenas alcanzan para una de 60.

### 3.10. Diagnóstico de inestabilidad temporal (rolling 60 días, ex "cambios estructurales")

Periodos marcados en zona "extremo" (percentil ≥90 histórico del rolling de
la mediana de `future_max_excursion_pts_60m`):

```
2022 Q1-Q3 · 2025 Q1-Q3 (con 2025Q2 el más extenso, 55 días) · 2026 Q1-Q2
```

Coincide cualitativamente con los periodos señalados en la notebook histórica
(2022, 2025Q2, 2026), reproducido aquí como diagnóstico exploratorio
explícitamente etiquetado (nunca "cambio estructural").

### 3.11. Dependencia temporal (ACF, Ljung-Box gap-aware, ARCH-LM), global, `full_day_eligible`, n=1.022.580

**Recalculado tras la validación focalizada (ver §7): Ljung-Box ahora se
calcula de forma gap-aware** (nunca forma un par entre el último valor de un
segmento y el primero del siguiente). Cifras vigentes:

```
                 Ljung-Box gap-aware (10 lags)    ARCH-LM (10 lags, no gap-aware)
ret_1m           stat=248,0   p≈0                 stat=42.305,7   p≈0
abs_ret_1m       stat=1.140.166   p≈0             stat=42.305,7   p≈0
squared_ret_1m   stat=75.172,0   p≈0              stat=273,2      p=7,1e-53
```

Pares efectivos usados en el lag 10 (gap-aware): **1.007.760**, frente a los
`n-k=1.022.570` que asumiría la fórmula clásica sobre una serie sin
interrupciones (diferencia: 14.810 pares transfronterizos excluidos,
consistente con los 1.481 límites de segmento reales × ~10 lags). El cambio
de magnitud del estadístico frente al cálculo previo (no gap-aware) es
pequeño (+2,0% en `ret_1m`, +0,40% en `abs_ret_1m`, +0,38% en `squared_ret_1m`)
y **no cambia ninguna conclusión** (los tres p-values siguen ≈0). Detalle
completo, incluyendo magnitudes de ACF por lag y comparación por régimen, en
§7.3.

Dependencia serial significativa en las tres series, y evidencia fuerte de
heterocedasticidad condicional (ARCH) tanto en `ret_1m` como en `abs_ret_1m`.
Por régimen, ARCH-LM sobre `ret_1m` es significativo en los 5 regímenes
(estadísticos entre 407 y 48.704, todos `p<10⁻⁸⁰`). Esto es **diagnóstico**:
no se usó para seleccionar ninguna feature. **ARCH-LM no es gap-aware**
(hereda la misma limitación que tenía Ljung-Box antes de esta corrección,
ver §7.3) — no se corrigió en este pase por estar fuera del alcance
solicitado, y queda documentado en §6.

### 3.12. Correlación OHLCV (`full_day_eligible`, Pearson)

Redundancias esperadas y confirmadas: `ret_1m`≈`log_ret_1m` (r=0,99999),
`delta_close_1m`≈`body_signed_pts` (r=0,998), `abs_delta_close_1m`≈`body_abs_pts`
(r=0,997) — información útil para que S05 evite duplicar variables
prácticamente colineales.

### 3.13. Calidad numérica

Sin infinitos en ninguna métrica OHLCV ni de ventana. NaN solo en la primera
barra de cada segmento (1.482 para métricas de 1 minuto, consistente con
1.482 días/segmentos completos).

---

## 4. Conclusiones derivadas (interpretación, no solo el número)

- El **framework de ventanas histórico era sólido y se confirma empíricamente**:
  una vez corregidos regímenes y repunte de dataset, las cifras globales de
  validez y excursión por horizonte no cambiaron sobre `full_day_eligible`
  (coinciden con v1), porque los rollovers reales nunca caen dentro de un
  segmento de minutos consecutivos. El bloqueo por contrato único era una
  protección necesaria de todos modos (ver `02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md`
  §5.1 sobre rollover no auditado): ahora está verificado, no solo asumido.
- **Régimen `Closing` corregido cambia su peso relativo** (4.34%→8.83%) y
  expone una limitación estructural real (sin muestra a h=90, mínima a h=60)
  que S03/S04 deben tener en cuenta al definir thresholds/targets por régimen
  para `Closing` — no es un bug de S02, es información nueva para decisiones
  posteriores.
- **Régimen y contrato son las dimensiones más inestables**, no el año — esto
  es consistente con la advertencia histórica de tratar el dataset por
  contexto, y da una base cuantitativa (CV) donde antes solo había
  observación cualitativa.
- **Hay dependencia serial y ARCH detectables** en los retornos de 1 minuto,
  global y en todos los regímenes — información relevante para S05 (features
  de volatilidad/momentum tienen sustento estadístico previo), pero **no se
  interpreta aquí como señal predictiva explotable**: es un diagnóstico de
  las propiedades de la serie, no una prueba de que un modelo pueda
  aprovecharlas después de costos y ejecución. Ljung-Box se corrigió para ser
  gap-aware (§7.3): el efecto sobre la magnitud del estadístico fue pequeño
  (<2%) y no altera esta conclusión.
- 30/60/90 minutos **siguen siendo candidatos razonables**: la evidencia
  exploratoria a favor de h60 como compromiso entre magnitud y disponibilidad
  se reproduce cuantitativamente, pero sigue siendo evidencia exploratoria,
  no una decisión de horizonte definitivo (eso corresponde a S03/S04).

---

## 5. Supuestos

- `window_segment_id` se recalculó sobre la población ya filtrada (no se
  reutilizó `consecutive_segment_id` de S01 directamente) para
  `partial_regime_eligible`, porque un segmento original de S01 puede
  contener regímenes elegibles no adyacentes tras el filtrado. Para
  `full_day_eligible` ambos coinciden por construcción (ningún bar se
  descarta).
- ACF y Ljung-Box (tras la corrección de §7.3) se calculan de forma
  gap-aware: los productos cruzados de la ACF solo se forman dentro de un
  mismo `consecutive_segment_id`, nunca entre el último valor de un segmento
  y el primero del siguiente. **ARCH-LM (`statsmodels.stats.diagnostic.het_arch`)
  sigue sin ser gap-aware** — arma su matriz de regresores lageados sobre el
  array completo sin conocer límites de segmento; no se corrigió en este
  pase (fuera del alcance solicitado), queda documentado en §6.
- El umbral mínimo de muestra para diagnóstico de dependencia por régimen
  (`min_obs_regime=5000`) se cumplió en los 5 regímenes de `full_day_eligible`
  (rango real: 88.920–400.140 observaciones); no se probó el caso de
  insuficiencia real con datos reales, solo sintéticamente (unit tests).

---

## 6. Limitaciones pendientes (heredadas, no resueltas ni corresponde resolverlas aquí)

```
Zona horaria de origen: sigue sin confirmación documental del proveedor (heredado de S00/S01).
timestamp_semantics: sigue unknown_not_confirmed (heredado de S01).
244 jornadas partial_undetermined + 11 no_data_undetermined: sin causa determinada (heredado de S01).
Auditoría general de rollover (Fase 4 del rebuild plan): S02 v2 solo aporta el impacto
  a nivel de ventana (0 transiciones intra-segmento); la auditoría completa de las 25
  transiciones crudas (gap de precio, contexto) sigue pendiente como etapa separada.
ARCH-LM no es gap-aware a nivel de lag individual (ACF y Ljung-Box ya se corrigieron, ver §7.3).
```

---

## 7. Validación focalizada (post-aprobación inicial)

Tras la primera entrega, el usuario pidió verificar tres posibles
inconsistencias antes de aprobar. Resultado de cada una:

### 7.1. Validez de ventanas — ¿la frase mezcla `window_type`?

Se extrajo `s02_window_validity_summary.parquet` completo para
`full_day_eligible` y se comparó cada fila contra la fórmula teórica para
691 barras/día × 1.482 días:

```
hist(h)   esperado = 1.482 × (691 − h + 1)
future(h) esperado = 1.482 × (691 − h)
full(h)   esperado = 1.482 × (691 − 2h + 1)
```

**Las 9 filas coinciden exactamente con la fórmula (diferencia = 0 en
todas)**: el artefacto está bien calculado y bien etiquetado por
`window_type`. No hay error de cálculo ni de etiquetado.

El problema estaba únicamente en la **redacción del resumen de cierre de la
entrega anterior** ("95.80% / 91.46% / 82.78% para 30/60/90 minutos"), que
combinaba sin decirlo tres `window_type` distintos:

```
95,80% = historical, h=30
91,46% = historical, h=60  (numéricamente igual a full, h=30 — coincidencia
                            algebraica: hist(h)=full(h/2) cuando h es par)
82,78% = full, h=60         (NO historical, h=90, que es 87,12%)
```

**Conclusión: error de redacción, no de código ni de artefacto.** Se
corrigió el texto (§3.6 ya usa la tabla 3×3 completa, sin resúmenes de una
sola línea que mezclen `window_type`).

### 7.2. `Closing` y horizonte de 90 minutos — ¿hay una restricción implícita de régimen?

Se revisó `build_window_validity` línea por línea: la función nunca lee
`regime_id` (confirmado por `grep`); las únicas condiciones son `date`
(implícita en `window_segment_id`), `window_segment_id`, `contract_run_id` y
consecutividad de `seg_bar_pos`. **No existe ninguna restricción implícita de
régimen.**

Cifras exactas para `regime_id=4` (`Closing`, `full_day_eligible`, h=90):

```
Barras Closing totales:            90.402
Ventanas históricas válidas (h=90): 90.402  (100%)
Ventanas futuras válidas (h=90):         0  (  0%)
Ventanas full válidas (h=90):            0  (  0%)

hist_invalid_reason_90m:   100% "valid"
future_invalid_reason_90m: 100% "insufficient_bars"  (nunca "contract_change")
```

Ejemplos de ventanas históricas válidas (timestamps reales,
America/New_York): `2020-01-02 15:00:00`, `2020-01-02 15:01:00`,
`2020-01-02 15:02:00`.

`Closing` ocupa las posiciones 630–690 dentro del segmento de 691 barras del
día. Para h=90, la ventana futura exigiría llegar hasta la posición
`seg_bar_pos + 90 ≤ 690`, es decir `seg_bar_pos ≤ 600` (minuto ≤ 870) — ningún
minuto de `Closing` (900–960) cumple esa condición. La causa exacta,
confirmada por conteo exhaustivo, es **`insufficient_bars`**: el día termina
antes de que se completen los 90 minutos futuros, no una restricción de
régimen. Esto coincide con lo ya reportado en §3.9; esta sección deja el
hallazgo verificado con conteos exactos y ejemplos, no solo descrito.

### 7.3. ACF y Ljung-Box — ¿cruzan pares entre segmentos?

**Confirmado: sí cruzaban.** `_clean_series` (implementación previa a esta
corrección) hacía `serie.replace([inf,-inf], nan).dropna()` y pasaba el
array resultante directamente a `statsmodels.tsa.stattools.acf(fft=True)`.
Como cada segmento solo tiene un NaN (su primera barra), el `dropna()`
**compactaba** el array eliminando esa fila pero dejando **adyacentes en el
array** al último valor de un segmento y al segundo valor (primero no-NaN)
del segmento siguiente — dos observaciones que en la realidad pueden estar
separadas por una noche entera o un fin de semana. `statsmodels.acf` no tiene
forma de saber esto: trata cualquier par de posiciones consecutivas del
array como un par lag-1 válido.

Cuantificación exacta sobre `full_day_eligible` (`ret_1m`, n=1.022.580,
1.482 segmentos): **1.481 de los 1.022.579 pares lag-1 (0,14%) cruzaban un
límite de segmento**; a lag=10, **14.810 de 1.022.570 pares (1,45%)** eran
transfronterizos.

**Corrección aplicada:** nuevas funciones `_clean_series_with_segment`,
`_segment_blocks`, `_gap_aware_acf` y `_ljung_box_gap_aware` en
`src/data/s02_intraday_analysis.py`. La media y la varianza total (`c0`) se
calculan sobre todas las observaciones válidas (son reductores escalares, no
pares); los productos cruzados numerador de cada lag se acumulan **solo**
dentro de cada `consecutive_segment_id`, sumando después numeradores y
denominadores de todos los segmentos sin crear ningún par entre ellos. El
estadístico de Ljung-Box usa el conteo **real** de pares por lag
(`pares_efectivos_k`, sumado por segmento) en vez de `n−k` (que asume
falsamente una serie sin interrupciones).

**Antes vs. después (global, `full_day_eligible`, `ret_1m`):**

| lag | ACF antes | ACF después | pares efectivos (después) | pares naive `n−k` |
|---:|---:|---:|---:|---:|
| 1 | 0,010973 | 0,010925 | 1.021.098 | 1.022.579 |
| 5 | -0,003863 | -0,003968 | 1.015.170 | 1.022.575 |
| 10 | 0,002982 | 0,002777 | 1.007.760 | 1.022.570 |

**Antes vs. después (global, `full_day_eligible`, `abs_ret_1m`):**

| lag | ACF antes | ACF después | pares efectivos (después) |
|---:|---:|---:|---:|
| 1 | 0,365630 | 0,365702 | 1.021.098 |
| 5 | 0,335161 | 0,334634 | 1.015.170 |
| 10 | 0,317518 | 0,316277 | 1.007.760 |

**Ljung-Box global (10 lags):**

| serie | Q antes | Q después (gap-aware) | p-value (ambos) |
|---|---:|---:|---|
| `ret_1m` | 243,14 | 247,98 | ≈0 |
| `abs_ret_1m` | 1.135.673 | 1.140.166 | ≈0 |
| `squared_ret_1m` | 74.884,30 | 75.172,02 | ≈0 |

**Por régimen (`ret_1m`, Ljung-Box 10 lags):**

| régimen | Q antes | Q después | pares efectivos lag=10 (después) | n_obs |
|---|---:|---:|---:|---:|
| Closing | 52,84 | 60,01 | 75.582 | 90.402 |
| Early_Premarket | 96,43 | 102,70 | 339.378 | 354.198 |
| Opening | 79,18 | 75,86 | 74.100 | 88.920 |
| Premarket | 38,75 | 42,35 | 74.100 | 88.920 |
| Regular | 141,22 | 136,55 | 385.320 | 400.140 |

**Magnitud del efecto:** cambios de ACF entre 0,00001 y 0,0012 en valor
absoluto; cambios de Q por régimen entre -4,20% (Opening) y +13,55% (Closing)
— mayor efecto relativo en regímenes con menos barras por segmento, como
`Closing`, donde cada segmento diario aporta solo ~61 observaciones y el
recorte de pares transfronterizos (1.481 en todo el dataset, uno por día)
pesa proporcionalmente más sobre una muestra por-régimen más chica. **En
ningún caso cambia la conclusión**: los 15 p-values (global + 5 regímenes ×
3 series) siguen siendo altamente significativos (`p<10⁻⁵`) antes y después.

`ARCH-LM` (`het_arch`) **no se modificó**: comparte la misma limitación
estructural (arma su matriz de regresores lageados ignorando límites de
segmento), pero corregirlo requeriría reimplementar la prueba manualmente
(no es una operación de reescalado como la ACF), lo cual excede el alcance
solicitado en este pase. Se verificó que sus resultados son **idénticos**
antes/después de esta corrección (no se tocó su código) y queda declarado
como limitación pendiente en §6.

**Pruebas nuevas:** 5 pruebas unitarias (`test_gap_aware_acf_excludes_cross_segment_pair`,
`test_gap_aware_acf_pairs_count_matches_sum_of_within_segment_pairs`,
`test_ljung_box_gap_aware_uses_effective_pairs_not_n_minus_k`,
`test_clean_series_with_segment_preserves_alignment_after_dropna`,
`test_acf_summary_reports_effective_pairs_lower_than_naive_n_minus_k_at_day_boundaries`)
verifican explícitamente que ningún par se forma entre segmentos distintos.

---

## 8. Estado y recomendación

Todos los criterios de aprobación acordados se verificaron por ejecución real:
reproducible desde kernel limpio, sin rutas rotas ni artefactos v1, régimen
vigente, `body_signed_pts`/`body_abs_pts` sin ambigüedad, ventanas validadas
por día **y por contrato**, terminología de inestabilidad temporal correcta,
generador de features exploratorio ausente del pipeline oficial, sin
afirmaciones de holdout ciego 2025–2026, sin construcción de targets.

La validación focalizada (§7) encontró un problema real y lo corrigió (ACF/
Ljung-Box gap-aware), y descartó los otros dos como falso positivo de
redacción (validez de ventanas) y comportamiento correcto y ya documentado
(`Closing`/h=90, causa exacta ahora confirmada con conteos y ejemplos). El
efecto de la corrección sobre los resultados numéricos es pequeño (<2% en
Ljung-Box global, hasta 13,6% en el régimen con menor muestra por segmento) y
no cambia ninguna conclusión reportada. ARCH-LM comparte la misma limitación
gap-aware y queda pendiente, declarada explícitamente, sin bloquear la
aprobación (es diagnóstico exploratorio, no usado para features).

**Estado: APPROVED.**
