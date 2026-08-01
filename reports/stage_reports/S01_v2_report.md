# S01 v2 — Reporte de preparación intradía MNQ

Implementación autónoma aprobada tras la auditoría/diseño de S01 v2 (ventana,
regímenes, validación de zona horaria, calendario híbrido, tratamiento de
gaps de S00). No se tocó S00, `config/data_config.yaml`, la notebook S01
original, S02+, ni archivos fuente. Sin commit ni push.

---

## 1. Archivos creados y modificados

**Creados:**

```
config/intraday_config.yaml
src/data/s01_intraday_preparation.py
tests/test_s01_intraday_preparation.py   (26 pruebas unitarias)
tests/test_s01_integration.py            (16 pruebas de integración)
notebooks/S01_intraday_data_preparation_v2.ipynb
data/02_intraday/mnq_intraday_v2.parquet
data/02_intraday/mnq_intraday_v2_summary.json
data/02_intraday/mnq_intraday_v2_manifest.json
data/02_intraday/trading_day_audit_v2.parquet
data/02_intraday/regime_distribution_v2.parquet
data/02_intraday/tz_validation_v2.json
reports/stage_reports/S01_v2_report.md
```

**Modificados:** ninguno (S00, `config/data_config.yaml`, la notebook S01
original y S02+ permanecen intactos).

---

## 2. Pruebas ejecutadas y resultados

| Suite | Alcance | Resultado |
|---|---|---|
| `tests/test_s01_intraday_preparation.py` | 26 unitarias, datos sintéticos (límites de régimen exactos incluyendo los 9 puntos pedidos, ausencia de ruta default, segmentos consecutivos, clasificación de jornadas por escenario sintético, hipótesis de zona horaria con un pico sintético conocido, chequeo DST, staleness) | **26/26 PASSED** |
| `tests/test_s01_integration.py` | 16 de integración sobre `mnq_raw_v2.parquet` real, marcadas `@pytest.mark.integration`, escritura exclusiva en `tmp_path` | **16/16 PASSED** (~40s) |
| Total S01 | `pytest tests/test_s01_*.py -v` | **42/42 PASSED** |
| Suite completa del repo | `pytest -v` | 76/77 PASSED. La única falla es `tests/test_s00_integration.py::test_never_writes_to_productive_raw_dir`, **preexistente y no relacionada con este trabajo**: ese test de S00 asume `data/01_raw/` vacío, pero ya contiene los artefactos productivos de S00 v2 generados y aprobados en una sesión anterior. No se tocó ningún archivo de S00 (`git diff --stat` sobre `tests/test_s00_integration.py` y `src/data/s00_raw_ingestion.py` está vacío). |

Durante el desarrollo se encontró y corrigió un problema real de round-trip
en Parquet: la columna `missing_minutes` (lista de enteros) se releía como
`numpy.ndarray` en vez de `list`, lo que rompía la comparación de igualdad
lógica de `atomic_write_parquet` (función de S00, no modificable). Se
resolvió serializando `missing_minutes` como JSON-string dentro de
`s01_intraday_preparation.py`, sin tocar código de S00.

---

## 3. Validación comparativa de zona horaria (UTC / America/New_York / America/Chicago)

No se asumió UTC. Se evaluaron programáticamente las 3 hipótesis contra
apertura 09:30 ET, cierre 16:00 ET y corte de mantenimiento ~17:00 ET,
separando meses EDT/EST, sobre las 2.172.640 filas de `mnq_raw_v2.parquet`:

| timezone_candidate | opening_alignment | closing_alignment | maintenance_break_alignment | dst_consistency | total_alignment_score | confidence_level |
|---|---:|---:|---:|---:|---:|---|
| **UTC** | 1 min | 0 min | 1 min | 0 min | **2.0** | **high** |
| America/Chicago | 29 min | 29 min | 999 (sin corte detectado) | 30 min | 1087.1 | low |
| America/New_York | 38 min | 59 min | 999 (sin corte detectado) | 9 min | 1105.1 | low |

UTC gana por un margen enorme (~500x mejor score que las alternativas) y de
forma consistente entre EDT y EST (`dst_consistency=0`). Las otras dos
hipótesis, al localizar directamente el índice tz-naive como si ya fuera
hora local, no logran ubicar el corte de mantenimiento dentro de la ventana
de búsqueda en absoluto (`999`) y además producen 1 fila ambigua/inexistente
por transición DST (localizar directamente en una zona con DST sí genera
ese problema; UTC, al no observar DST, no lo tiene).

## 4. Zona horaria seleccionada y evidencia

```
timezone_selected: UTC
timezone_validation_status: empirically_supported
timezone_provider_confirmation: false
timezone_evidence: inferred_from_market_structure_and_dst
confidence_level: high
```

**No se presenta como confirmada documentalmente** (no hay config de
exportación del proveedor) — es evidencia de comportamiento de mercado,
mucho más robusta que la anécdota de un solo día de S01 v1, pero sigue
siendo inferencia, no confirmación. Gráfico del perfil de volumen por minuto
local con marcadores en 08:30/09:30/15:00/16:00 disponible en la notebook
v2 (celda 8) y en `tz_validation_v2.json`.

---

## 5. Shape, columnas y rango de `mnq_intraday_v2.parquet`

```
Shape: (1.087.777, 11)
Columnas: date, minute_of_day, regime_id, regime_label, consecutive_segment_id,
          open, high, low, close, volume, contract
Índice: DatetimeIndex, America/New_York, tz-aware
Rango: 2019-12-23 04:30:00-05:00 -> 2026-04-17 16:00:00-04:00
minute_of_day: 270-960 (04:30-16:00), sin ningún valor fuera de rango
SHA-256: c8001e8e5660a1d77d0c84ccc472bba49ba7ca9f5d137a4c700278d3810c151c
```

A diferencia de S01 v1 (que descartaba todo día incompleto), v2 **conserva
todas las barras observadas** dentro de la ventana — de ahí que tenga más
filas que el histórico (1.024.062), no menos.

---

## 6. Jornadas completas e incompletas

```
Total fechas calendario auditadas (trading_day_audit_v2.parquet):        2.309
Jornadas con exactamente 691 barras (full_coverage):                     1.482
Jornadas parciales, CON datos (1-690 barras):                              305
Fechas SIN ningún dato observado:                                          522
```

Desglose correcto de las 827 fechas que no son `full_coverage` (827 = 305 + 522,
**no** son "827 jornadas incompletas" homogéneas — son dos poblaciones muy
distintas que no deben mezclarse):

```
305 jornadas parciales CON datos:
  244  partial_undetermined
   57  partial_early_close_cme
    4  partial_gap_documented_s00

522 fechas SIN ningún dato:
  475  no_data_weekend
   25  no_data_gap_documented_s00
   11  no_data_cme_holiday
   11  no_data_undetermined
```

**Verificación cruzada exacta con S01 v1:** `1.482 × 691 = 1.024.062`,
idéntico al shape histórico de `mnq_intraday.parquet`. El subconjunto
`full_coverage` de v2 **reproduce exactamente** el dataset que v1 producía.
**Las 63.715 filas adicionales de v2 respecto de v1 (1.087.777 − 1.024.062)
provienen íntegramente de las 305 jornadas parciales con datos** (suma de
`observed_bars` sobre esas 305 fechas = 63.715, verificado), **no** de las
827 fechas no-`full_coverage` en conjunto — las 522 fechas sin datos
aportan, por definición, 0 filas.

Distribución de `day_status` (2.309 fechas):

```
full_coverage                 1.482
no_data_weekend                  475
partial_undetermined             244
partial_early_close_cme           57
no_data_gap_documented_s00        25
no_data_cme_holiday               11
no_data_undetermined              11
partial_gap_documented_s00         4
```

Distribución de `missing_bars` sobre las 305 jornadas con datos parciales
(>0 y <691 barras): media 482, mediana 689, min 1, max 690 — es decir, la
mayoría de las jornadas "parciales" están casi vacías (perdieron casi toda
la ventana) o casi completas (falta 1 barra), con relativamente pocas en un
punto intermedio.

---

## 7. Consecutividad y cobertura por régimen

`trading_day_audit_v2.parquet` incluye, por cada uno de los 5 regímenes y
cada una de las 2.309 fechas: `regime_N_observed_bars`,
`regime_N_expected_bars`, `regime_N_missing_bars`, `regime_N_coverage_ratio`,
`regime_N_is_consecutive`. Esto permite a S02+ evaluar elegibilidad a nivel
de régimen, no solo de jornada completa — es lo que sustenta la categoría
`partial_regime_eligible` (119 jornadas, ver §8).

---

## 8. Categorías de elegibilidad

```
full_day_eligible          1.482   (691/691, totalmente consecutivo)
not_model_eligible           526   (sin datos, o gap documentado de S00)
descriptive_only             182   (datos parciales, ningún régimen completo/consecutivo)
partial_regime_eligible      119   (jornada incompleta, pero ≥1 régimen completo y consecutivo)
```

Ningún día se clasificó únicamente por el número total de barras: una
jornada con datos parciales puede ser `partial_regime_eligible` si al menos
un régimen (p. ej. `Regular`) está completo y sin gaps internos, aunque el
resto de la jornada esté vacío.

---

## 9. Distribución de barras por `regime_id`

```
0  Early_Premarket   383.822 barras   1.626 días con ≥1 barra
1  Premarket          96.151 barras   1.614 días con ≥1 barra
2  Opening            94.746 barras   1.639 días con ≥1 barra
3  Regular           420.074 barras   1.692 días con ≥1 barra
4  Closing            92.984 barras   1.561 días con ≥1 barra
```

Suma = 1.087.777, igual al total de filas del dataset (verificado por
prueba automatizada). Ningún régimen recibió una barra por ruta `default`
(bug histórico corregido y verificado: la barra de 16:00 pertenece a
`Closing` en el 100% de los casos donde existe).

---

## 10. Tratamiento de DST

Verificación explícita (no solo argumentada como en v1) sobre las 14 fechas
de transición DST (2020-2026): **0 fechas con minutos duplicados** dentro de
la ventana 04:30-16:00. `tz_localize("UTC").tz_convert("America/New_York")`
confirmado seguro frente a DST para este dataset.

---

## 11. Tratamiento de los gaps de S00

Los 29 registros con `s00_gap_reference` no son "29 días perdidos": son
**29 fechas calendario cuyo rango se solapa con los intervalos de gap
documentados en S00** (`mnq_raw_v2_gaps.parquet`), incluidos los extremos
donde el gap empieza o termina a mitad de sesión y por tanto sí hay algo de
cobertura parcial ese día. Desglose exacto por `gap_id`:

| gap_id | fechas | `calendar_status=cme_trading_day` | `calendar_status=weekend` | con datos parciales (`observed_bars>0`) | sin datos (`observed_bars=0`) | `day_status` |
|---|---:|---:|---:|---:|---:|---|
| `s00_gap_M23` | 12 | 8 | 4 | 2 | 10 | 10 × `no_data_gap_documented_s00`, 2 × `partial_gap_documented_s00` |
| `s00_gap_H25_M25` | 17 | 11 | 6 | 2 | 15 | 15 × `no_data_gap_documented_s00`, 2 × `partial_gap_documented_s00` |
| **Total** | **29** | **19** | **10** | **4** | **25** | 25 × `no_data_gap_documented_s00`, 4 × `partial_gap_documented_s00` |

Ninguna de las 29 fechas es `no_data_cme_holiday` — todas caen en fechas que
`CME_Equity` considera día de trading (19) o fin de semana (10), nunca
feriado, porque los rangos de gap fueron definidos en S00 a partir de los
timestamps reales de los archivos fuente, no del calendario. Los 2 extremos
parciales de cada gap (`2023-04-05` con 574 barras y `2023-04-16` con 1
barra para M23; `2025-03-21` con 301 barras y `2025-04-06` con 1 barra para
H25→M25) corresponden exactamente al primer/último timestamp observado
antes/después de cada gap en S00 — es decir, son la sesión que el gap
interrumpe a mitad de camino, no un error de clasificación.

Ningún dato fue rellenado, interpolado ni eliminado. Las 29 fechas quedan
marcadas `day_status ∈ {no_data_gap_documented_s00, partial_gap_documented_s00}`,
`is_model_eligible=false`, con `s00_gap_reference` apuntando al `gap_id`
correspondiente — trazabilidad completa hacia `mnq_raw_v2_gaps.parquet`.

**Patrón recurrente 16:20-16:30 (2019-2021):** cae **fuera** de la ventana
primaria `04:30-16:00` (el patrón ocurre después de las 16:00), por lo que
**no afecta directamente a `mnq_intraday_v2.parquet`** ni a ninguna fila,
régimen o jornada de este dataset. Queda documentado en
`config/intraday_config.yaml → known_gaps` como `s01_pattern_1620_1630`,
`blocking: false`, únicamente como hallazgo relevante para el día en que se
evalúe extender la ventana operativa más allá de las 16:00 — no como un
problema pendiente de S01 v2 en su forma actual.

---

## 12. Comparación documental con S01 v1

| Campo | S01 v1 (histórico) | S01 v2 (verificado) |
|---|---|---|
| Calendario | `NASDAQ` (equivocado para un futuro CME Globex) | Híbrido: `CME_Equity` como referencia + datos observados como evidencia principal |
| Filas totales | 1.024.062 (solo días 100% completos) | 1.087.777 (todas las barras observadas en la ventana) |
| Subconjunto "691/691" | 1.482 días × 691 = 1.024.062 | **Idéntico**: 1.482 días × 691 = 1.024.062 |
| Zona horaria | Asumida UTC sin comparar alternativas | UTC seleccionada tras comparar 3 hipótesis (score 2.0 vs 1087/1105) |
| Régimen 16:00 | `Overnight` (bug, `default=0`) | `Closing` (`regime_id=4`), sin ruta default |
| Régimen 0 (04:30-08:29) | `Overnight` | `Early_Premarket` (mismo `regime_id=0`, etiqueta corregida) |
| Días no `full_coverage` | Eliminados sin clasificar (80 de 1.562, calendario NASDAQ) | Conservados y clasificados: 305 parciales con datos + 522 fechas sin datos, de 2.309 (calendario CME_Equity + evidencia observada) |
| Gaps de S00 | No existían como concepto (S00 v1 no los documentaba) | 29 días vinculados explícitamente a `mnq_raw_v2_gaps.parquet` |
| Gobernanza | `load_or_build` sin hash/versión | Manifest autoritativo con hash de fuente+módulo+config, staleness, git como metadata |

---

## 13. Problemas pendientes

1. `timestamp_semantics` sigue `unknown_not_confirmed` — no se encontró evidencia suficiente (ni en S00 ni en S01) para determinar si el timestamp marca inicio o cierre de barra. No se desplazó ninguna barra.
2. `timezone_provider_confirmation: false` — la selección de UTC es evidencia empírica robusta, no confirmación documental del proveedor.
3. 244 jornadas `partial_undetermined` y 11 `no_data_undetermined` — CME_Equity las marca como día de trading pero la cobertura observada no calza con ningún patrón conocido (ni gap de S00, ni cierre anticipado documentado). Quedan para auditoría futura, no bloquean S01 v2.
4. El patrón recurrente 16:20-16:30 (2019-2021) sigue sin investigarse a fondo — cae fuera de la ventana primaria y no afecta `mnq_intraday_v2.parquet`; solo relevante si en el futuro se extiende la ventana después de las 16:00.
5. La discrepancia entre `CME_Equity` y `"CME Globex Equity"` (1 día distinto, 2025-01-09) no se resolvió — se usó `CME_Equity` según lo aprobado, sin revalidar la alternativa.

---

## 14. Recomendación de aprobación o rechazo

**Recomiendo aprobar S01 v2.** Las 42 pruebas propias pasan, la validación
de zona horaria es una comparación programática real (no una hipótesis
asumida), el corte de mantenimiento CME se localizó independientemente en
S00 y se re-confirmó aquí con la misma precisión, el subconjunto
`full_coverage` reproduce exactamente el resultado histórico de S01 v1
(verificación cruzada exacta: 1.482×691=1.024.062), los dos bugs críticos de
régimen quedaron corregidos y verificados con pruebas de regresión, y los
gaps de S00 quedan trazados explícitamente en vez de perderse en la regla
de exclusión por conteo. Los problemas pendientes (§13) son limitaciones de
evidencia, no defectos de esta implementación, y están declarados
explícitamente en config/manifest/summary.

---

## 15. `git status --short` y `git diff --stat`

Ver salida siguiente en la entrega de este turno (comandos ejecutados justo
antes de este reporte, nada fue commiteado).

---

## 16. Addendum — Resolución de rollover y cierre anticipado verificado (2026-07-31)

Esta revisión agrega a S01 v2 dos piezas que faltaban respecto al cierre
original de este reporte (§1-§15, que sigue vigente en lo demás):

### 16.1. Resolución de rollover (nuevo)

El cierre original de S01 v2 no seleccionaba contrato: el dataset podía
tener más de una barra por minuto en fechas de rollover con solapamiento.

**Conteo verificado de transiciones** (artefacto:
`reports/stage_reports/s01_rollover_transition_audit.csv`, una fila por
par de contratos consecutivos, generado programáticamente a partir de
`data/01_raw/mnq_raw_v2.parquet` — no a mano):

```text
27 contratos en el historial completo (H20 .. U26)  ->  26 transiciones posibles
  23 transiciones son handoffs limpios: CERO fechas con ambos contratos
     presentes (verificado por interseccion de fechas por contrato)
   3 transiciones tienen solapamiento real, confirmado por volumen:
     Z24->H25, H25->M25, M26->U26
```

Una corrección importante respecto a un borrador previo de este informe:
la cifra correcta de transiciones es **26** (27 contratos − 1), no 25; y
los handoffs limpios son **23**, no 22. El número "25" que aparecía antes
mezclaba, por error, el total de *transiciones* con el total de *filas* en
`rollover_ambiguous_dates_v2.parquet` (que sí es 25, ver abajo) — son dos
conteos distintos que coinciden por casualidad en ser cercanos.

`rollover_ambiguous_dates_v2.parquet` registra **25 filas** (fechas
evaluadas por el algoritmo), no confundir con las 26 transiciones:

```text
23 filas: fecha con los DOS contratos presentes simultaneamente ese dia
           (7 en Z24/H25, 7 en H25/M25, 9 en M26/U26)
 2 filas: fecha donde el contrato activo tiene 0 barras pero cae DENTRO de
           una ventana de solapamiento aun no confirmada (2025-03-16 y
           2025-03-17, dentro de la ventana H25->M25) -- no son un
           solapamiento de 2 contratos ese dia especifico, pero tampoco
           son un handoff limpio: se evaluan con la misma logica
           conservadora (no confirman, no se rellenan).
```

`resolve_rollovers` (`src/data/s01_intraday_preparation.py`) construye la
serie principal con exactamente un contrato por fecha, sin promediar
OHLCV ni crear barras sintéticas, aplicando: detección de fechas
ambiguas, comparación de volumen solo sobre minutos compartidos,
confirmación exclusivamente con sesión compartida completa (691/691) y
≥55% de volumen del entrante, activación desde la jornada siguiente
observada (nunca la fecha de la señal), irreversibilidad, y trazabilidad
completa de las filas descartadas.

**Regresión verificada contra los datos reales** (`tests/test_s01_integration.py`):

```text
Z24 -> H25: senal 2024-12-17 (share 69.10%), H25 activo desde 2024-12-18
H25 -> M25: senal 2025-03-18 (share 69.09%), M25 activo desde 2025-03-19
M26 -> U26: senal 2026-06-15 (share 76.44%), U26 activo desde 2026-06-16
2025-03-15: NO confirma (sesion no completa: H25 2/691, M25 1/691)
2026-06-11: conserva M26 con su cobertura real (646/691); U26 (636/691,
  share 1.3%) no confirma
```

**Validación de conservación (bloqueante, en código productivo, no solo en
pruebas):** `resolve_rollovers` calcula `len(resolved_df) + len(discarded_df)`
y lo compara contra `len(df_window)` (la entrada, antes de resolver);
`build_manifest` repite el mismo chequeo y hace fallar la construcción del
manifest si no coincide. Resultado persistido en
`mnq_intraday_v2_manifest.json → rollover.conservation_check`:

```text
1.166.364 (filas antes de resolver rollover) = 1.151.817 (resueltas) + 14.547 (descartadas)
conservation_check_passed: true
```

Nuevos artefactos: `rollover_events_v2.parquet` (3 transiciones
confirmadas), `rollover_ambiguous_dates_v2.parquet` (25 filas evaluadas),
`rollover_discarded_rows_v2.parquet` (14.547 filas descartadas con
motivo, nunca borradas silenciosamente), `consecutive_segments_v2.parquet`
(resumen de segmento por fecha: inicio, fin, longitud),
`s01_rollover_transition_audit.csv` (26 transiciones, ver arriba) y
`s01_rollover_overlap_dates_full_table.csv` (tabla completa por fecha:
contratos, filas por contrato, volumen compartido, contrato
seleccionado/descartado, `day_status` antes/después y motivo — ver §16.4).

### 16.2. Clasificación de jornadas revisada

- `full_coverage` ahora exige también contrato único
  (`n_contracts_observed <= 1`), verificado con una invariante que hace
  fallar `build_trading_day_audit` si alguna fecha llega con más de un
  contrato — no puede volver a ocurrir silenciosamente.
- `partial_early_close_cme` exige la intersección de DOS condiciones
  independientes, no una sola: (a) la fecha está declarada como cierre
  anticipado en el calendario oficial versionado `pandas_market_calendars`
  (paquete externo mantenido, no inventado por este pipeline — ver §16.5
  para la verificación completa de las 40 fechas contra ese calendario) y
  (b) los datos observados calzan EXACTO con 511 barras consecutivas
  04:30-13:00 (`config/intraday_config.yaml: early_close`). Las 40 fechas
  que cumplen ambas condiciones quedan en una categoría de elegibilidad
  nueva, separada de `full_day_eligible`: `early_close_eligible` (no entra
  por defecto en la población principal).

### 16.3. Efecto de la actualización de datos de S00

`data/00_source` se actualizó (ver addendum §12 de `S00_v2_report.md`): 27
archivos en vez de 26, cobertura hasta 2026-07-31, y desaparición de los
gaps S00-05 (H25→M25) y S00-06 (M23 interno). Efecto combinado (fuente
actualizada + rollover resuelto + cierre anticipado verificado) sobre la
clasificación de las 2.414 fechas calendario del rango:

```text
day_status:
  full_coverage             1.570  (antes 1.482)
  no_data_weekend             509
  partial_undetermined        275
  partial_early_close_cme      40  (categoria antes inferida, ahora verificada)
  no_data_cme_holiday          11
  no_data_undetermined          9
  no_data_gap_documented_s00    0  (antes 25; el gap ya no existe en los datos)
  partial_gap_documented_s00    0  (antes 4; el gap ya no existe en los datos)

eligibility_category:
  full_day_eligible          1.570
  partial_regime_eligible        87
  early_close_eligible           40  (categoria nueva)
  descriptive_only              188
  not_model_eligible            529
```

Estas cifras ya incluyen el efecto de la regla de respaldo 11 (§16.7,
agregada en una revisión posterior de este cierre): `full_coverage` sube
de 1.569 a 1.570 porque `2025-03-17` deja de perderse y pasa a contarse
como completa.

### 16.4. Tres estados de S01 (para no confundir shapes de distinto origen)

Auditoría de cierre solicitada explícitamente para separar qué cambió por
la actualización de fuente y qué cambió por la resolución de rollover:

| Estado | Filas | `full_coverage` | Cómo se obtuvo |
|---|---:|---:|---|
| **Artefacto histórico** (fuente pre-actualización, sin resolución de rollover) | 1.087.777 | 1.482 | Documentado en `01_CURRENT_DECISIONS.md §32` antes de esta revisión; fuente original ya no existe (fue reemplazada), no reproducible directamente. |
| **Datos actuales, INMEDIATAMENTE ANTES de resolver rollover** | 1.166.364 | 1.549 | Recalculado en esta auditoría: filtro de ventana sobre `mnq_raw_v2.parquet` actual, SIN `resolve_rollovers` y SIN el requisito de contrato único (clasificación al estilo anterior al fix, aplicada a los datos ya actualizados). |
| **Resultado final** (fuente actualizada + rollover resuelto, incluida la regla de respaldo 11 + contrato único + cierre anticipado verificado) | 1.152.510 | 1.570 | `data/02_intraday/mnq_intraday_v2.parquet` / `trading_day_audit_v2.parquet` vigentes. |

La fila intermedia aísla el efecto de la actualización de fuente por sí
sola (1.087.777→1.166.364 filas, 1.482→1.549 `full_coverage`, SIN
rollover); la fila final aísla además el efecto de resolver el rollover
(1.166.364→1.152.510 filas: se descartan 13.854 filas correspondientes a
un contrato no elegido — ver conservación exacta más abajo). El cambio
1.549→1.570 `full_coverage` (neto +21) ocurre **enteramente dentro de las
25 fechas ambiguas** de §16.1/§16.6 (verificado en
`s01_rollover_overlap_dates_full_table.csv`): 22 de esas 25 pasan a ser
`full_coverage` (21 porque ya no tienen un segundo contrato mezclado
bloqueando la clasificación, más `2025-03-17` gracias a la regla de
respaldo 11 — §16.7), y 1 (`2025-03-16`, con solo 2 barras reales de M25)
sigue sin ser `full_coverage` bajo cualquier lógica (22 − 1 = 21). Las
2.389 fechas restantes del rango no cambian de clasificación entre el
estado intermedio y el final, porque `resolve_rollovers` no las toca.

**Conservación exacta** (bloqueante en `resolve_rollovers` y en
`build_manifest`, no solo en pruebas):
`1.166.364 = 1.152.510 (resueltas) + 13.854 (descartadas)`.

### 16.5. Verificación de las 40 fechas `partial_early_close_cme` contra calendario oficial

**Fuente:** `pandas_market_calendars==5.4.0`, calendario `CME_Equity`
(clase `CMEEquityExchangeCalendar`, módulo `pandas_market_calendars.calendars.cme`)
— paquete externo versionado que modela el calendario de feriados y
cierres anticipados publicado por CME Group; no es una tabla escrita a
mano por este pipeline.

**Verificación ejecutada** (no solo el patrón de 511 barras): se llamó a
`get_cme_early_close_dates()` sobre el rango completo 2020-01-01 a
2026-12-31 (66 fechas de cierre anticipado declaradas por el calendario
en ese rango) y se comprobó que las 40 fechas clasificadas
`partial_early_close_cme` están **todas** presentes en ese calendario, con
hora de cierre declarada `13:00` en las 40. Lista completa (fecha, día de
semana, hora declarada por CME_Equity) persistida en
`reports/stage_reports/s01_early_close_dates_verified.csv` — 40 filas,
desde 2020-02-17 hasta 2026-07-03.

Las 40 fechas corresponden a los feriados de cierre anticipado
estándar de CME (MLK Day, Presidents Day, Memorial Day, Juneteenth, 3-5
de julio según caiga el 4, Labor Day, Thanksgiving), consistente con el
calendario público de CME Group. **Conclusión: las 40 fechas quedan
confirmadas** (no pendientes) porque están respaldadas por dos evidencias
independientes — calendario oficial versionado + patrón exacto de datos
— no solo por el patrón de 511 barras.

Nota de alcance: `pandas_market_calendars` es un paquete de terceros
mantenido activamente que modela el calendario publicado por CME, no una
consulta en vivo a una API oficial de CME Group; se documenta así para
que quien audite esta decisión sepa exactamente qué se verificó.

### 16.6. Tabla completa de fechas con solapamiento (`s01_rollover_overlap_dates_full_table.csv`)

25 filas (ver §16.1), con fecha, contratos, filas por contrato, volumen
compartido, contrato seleccionado/descartado, `day_status` antes
(recalculado con la lógica anterior al fix, sobre los datos actuales) y
`day_status` después (resultado final), motivo. **3 de las 25 fechas NO
quedan `full_coverage`** tras la resolución (`2025-03-17` sí queda
`full_coverage`, gracias a la regla de respaldo 11 — ver §16.7):

| Fecha | Contrato elegido | `day_status` final | `eligibility_category` | Motivo |
|---|---|---|---|---|
| 2025-03-15 | H25 | `partial_undetermined` | `descriptive_only` | Sesión compartida no completa (H25 solo 2 barras, M25 solo 1 esa fecha) — no puede confirmar ni clasificar como completa. |
| 2025-03-16 | M25 (regla de respaldo 11) | `partial_undetermined` | `descriptive_only` | H25 (activo) 0 barras; M25 (entrante) solo 2 barras — se conserva su cobertura real (no se rellena), pero 2/691 no alcanza para `full_coverage`. |
| 2026-06-11 | M26 | `partial_undetermined` | `partial_regime_eligible` | M26 (activo) tiene solo 646/691 barras ese día (no es una sesión completa); U26 tiene 636/691 con share de solo 1,3% — ninguna confirma. La fecha se clasifica por la cobertura REAL de M26. |

Pruebas: **28 unitarias + 40 de integración** de S01 pasan (68/68) contra
el dataset y la lógica actualizados — incluye pruebas de rollover
(regresión de las 3 transiciones confirmadas, casos límite, la regla de
respaldo 11 con el caso de regresión `2025-03-17`, irreversibilidad del
estado activo, conservación reportada en el manifest), cierre anticipado
verificado y unicidad de contrato; se removieron dos pruebas que
dependían de los gaps S00-05/S00-06 ya inexistentes, y se corrigió una
prueba de irreversibilidad para no confundir las fechas de respaldo con
una reversión real del contrato activo. Comando:
`pytest tests/test_s01_intraday_preparation.py tests/test_s01_integration.py -q`.

### 16.7. Regla de respaldo 11 — activo sin datos, entrante con datos válidos

**Regla añadida en una revisión posterior de este cierre.** Si el
contrato activo tiene EXACTAMENTE 0 barras una fecha, pero el contrato
entrante sí tiene datos esa fecha, se selecciona la cobertura real del
entrante para **esa fecha puntual únicamente** (motivo
`active_contract_no_data_fallback_to_incoming`). Condiciones aplicadas
(`resolve_rollovers` en `src/data/s01_intraday_preparation.py`):

```text
1. El contrato activo debe tener EXACTAMENTE 0 barras esa fecha.
2. El contrato entrante debe tener barras esa fecha (cualquier cantidad;
   la clasificacion final -- full_coverage o parcial -- la decide
   build_trading_day_audit con la logica estandar, no esta regla).
3. No se mezclan contratos: el activo no aporta nada ese dia.
4. No se crean ni completan barras sinteticas: se usa la cobertura real
   del entrante, completa o parcial.
5. Motivo registrado: active_contract_no_data_fallback_to_incoming.
6. NO adelanta formalmente el contrato activo para fechas siguientes: el
   cruce formal sigue dependiendo EXCLUSIVAMENTE de la confirmacion por
   volumen (reglas 3-4 del algoritmo de rollover).
7. Si el entrante tiene 691 barras consecutivas -> full_coverage normal.
8. Si tiene cobertura parcial -> clasificacion parcial estandar (sin
   relleno).
```

**Caso de regresión obligatorio, verificado:**

```text
2025-03-17: H25 (activo) 0 barras, M25 (entrante) 691 barras validas.
  -> S01 conserva M25 para esa fecha.
  -> day_status = full_coverage, eligibility_category = full_day_eligible.
  -> El rollover formal H25 -> M25 SIGUE confirmando el 2025-03-18,
     efectivo desde el 2025-03-19 (sin adelantarse).
```

`2025-03-16` (mismo mecanismo, pero M25 solo tiene 2 barras esa fecha)
también aplica la regla: se conservan las 2 barras reales de M25
(`day_status = partial_undetermined`, ya no `no_data_weekend` como antes,
porque ahora hay datos reales observados ese día).

**Efecto sobre las cifras del dataset** (respecto al estado sin esta
regla, documentado en §16.1-§16.6 antes de esta adición):

```text
full_coverage:        1.569 -> 1.570  (+1, 2025-03-17)
n_rows_resolved:       1.151.817 -> 1.152.510  (+693 = 2 + 691)
n_discarded_rows:      14.547 -> 13.854  (-693)
conservation_check:    1.166.364 == 1.152.510 + 13.854  (sigue pasando)
fechas NO full_coverage entre las 25 ambiguas: 4 -> 3
```

Ningún otro efecto: las 3 transiciones confirmadas (Z24→H25, H25→M25,
M26→U26) y sus fechas de señal/efectividad no cambiaron, porque la regla
11 nunca modifica el estado del contrato activo formal.
