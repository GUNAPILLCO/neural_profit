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
