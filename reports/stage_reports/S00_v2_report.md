# S00 v2 — Reporte de ingestión cruda MNQ

Generado tras la implementación aprobada en el plan de auditoría/reconstrucción de S00.
Cubre exclusivamente la etapa S00 (consolidación cruda). No toca S01 en adelante.

---

## 1. Archivos creados

```
environment/stage_02.yml
config/data_config.yaml
pytest.ini
src/__init__.py
src/data/__init__.py
src/data/s00_raw_ingestion.py
tests/test_s00_raw_ingestion.py         (25 pruebas unitarias)
tests/test_s00_integration.py           (10 pruebas de integración)
notebooks/S00_raw_data_preparation_v2.ipynb
manifests/s00_source_manifest.csv       (vista derivada, no autoritativa)
data/01_raw/mnq_raw_v2.parquet
data/01_raw/mnq_raw_v2_summary.json
data/01_raw/mnq_raw_v2_manifest.json    (fuente autoritativa de hashes/staleness)
data/01_raw/mnq_raw_v2_gaps.parquet
```

**No se tocó:** `notebooks/S00_raw_data_preparation.ipynb` (evidencia histórica intacta),
ningún archivo en `data/00_source/`, `data/01_raw/mnq_raw.parquet` /
`mnq_raw_summary.json` (no existían y no se crearon con esos nombres), S01 en
adelante, ni documentos de `mnq_project/`.

---

## 2. Comandos ejecutados

```bash
conda run -n stage_02 python -m pip install pytest pyyaml
conda run -n stage_02 python -m pytest -m "not integration" -v
conda run -n stage_02 python -m pytest -m integration -v
conda run -n stage_02 python -m pytest -v
conda run -n stage_02 jupyter kernelspec list
conda run -n stage_02 jupyter nbconvert --to notebook --execute --inplace \
    notebooks/S00_raw_data_preparation_v2.ipynb
```

Ninguna instalación fuera de `stage_02`; ningún commit; ninguna modificación
de la notebook original ni de S01.

---

## 3. Pruebas implementadas y resultado

| Suite | Alcance | Resultado |
|---|---|---|
| `tests/test_s00_raw_ingestion.py` | 25 pruebas unitarias, datos sintéticos en memoria (regex de nombre, extracción contract/instrument/contract_full, rechazo de esquema/timestamp/OHLC/volumen inválidos, archivo vacío, duplicados exactos vs. duplicados por (timestamp,contract) vs. mismo timestamp con contrato distinto, orden cronológico, clasificación estructural de gaps, hash de staleness, git como metadata no invalidante) | **25/25 PASSED** |
| `tests/test_s00_integration.py` | 10 pruebas sobre el corpus real de 26 archivos, marcadas `@pytest.mark.integration`, escritura exclusivamente en `tmp_path` | **10/10 PASSED** (~80s) |
| Total | `pytest -v` (sin filtro) | **35/35 PASSED**, 0 warnings |

Durante el desarrollo se encontró y corrigió un bug real en el chequeo de
"filas exactamente duplicadas": la primera versión comparaba solo las
columnas sin incluir el índice (`timestamp`), lo que habría marcado como
duplicadas dos barras distintas que coincidieran por azar en OHLCV. Se
corrigió incluyendo el índice en la comparación (`df.reset_index().duplicated()`)
y se añadió una prueba de regresión (`test_same_timestamp_different_contract_is_not_confused_with_duplicate`,
`test_duplicate_timestamp_different_ohlcv_caught_as_timestamp_contract_dup`).

---

## 4. Resultados de validación sobre los 26 archivos

Ejecución productiva completa (`force_rebuild` no fue necesario, primera
generación) sin ninguna excepción — es decir, **todas** las validaciones de
`config/data_config.yaml → validation.checks` (schema, parseo, timestamps,
monotonicidad, duplicados globales, duplicados por timestamp+contrato,
nulos, infinitos, precios positivos, volumen no negativo, volumen entero,
invariantes OHLC, filas exactamente duplicadas, archivos vacíos, transiciones
de contrato, gaps) pasaron sin necesidad de detener el pipeline:

```
Archivos fuente:        26 / 26 válidos (nombre, orden consecutivo 00-25, mes→H/M/U/Z)
Filas totales:           2.172.640
Filas rechazadas:        0
Columnas persistidas:    open, high, low, close, volume, contract  (idéntico al schema histórico)
Índice:                  DatetimeIndex, tz-naive
Rango:                   2019-12-23 03:01:00 → 2026-04-17 20:18:00
Duplicados exactos:      0
Duplicados (ts,contract):0
Gaps registrados:        4.243  (25 inter_contract + 4.218 intra_file)
  - structural_only:          3.498
  - unconfirmed:                 521
  - provisional_pattern_match:   224
```

Esto reconfirma, de forma independiente y con validaciones ahora persistidas
(no solo impresas), los mismos hallazgos de la auditoría inicial.

---

## 5. Manifiesto generado

`data/01_raw/mnq_raw_v2_manifest.json` (autoritativo) contiene, por archivo
fuente (26 registros): ruta, tamaño, fecha de modificación, SHA-256,
`instrument`/`contract`/`contract_full`, n_rows, primer/último timestamp.

Campos de staleness persistidos:

```
source_files_sha256:        {26 hashes}
module_sha256:               09fc5b32ecc44ada29f8586965b48679c4de39d95902e87967cddc4c1bd9afa1
config_sha256_normalized:    684ab3f6b5904bf480aa131658ed5f3e18b8292b214f7bad3295048b63ffbffc
schema_expected:             [datetime, open, high, low, close, volume, contract]
pipeline_version:            s00_v2
force_rebuild:                false
```

Metadata de procedencia (no invalidante):

```
git_commit: 3a42d413d573ce08a56ec35669071d61c7f72c4e
git_dirty:  true   (working tree tenía cambios sin commitear al generar el artefacto -- esperado, es esta misma implementación)
```

`manifests/s00_source_manifest.csv` es la vista tabular derivada (26 filas,
mismo contenido que `manifest.json → sources`, sin los agregados de gaps).

---

## 6. Resultados de los dos gaps extraordinarios

### Gap interno M23

| campo | valor |
|---|---|
| archivo/contrato | `13_mnq_06_23.Last.txt` (contrato `M23`) |
| tipo estructural | `intra_file` |
| extremo anterior | 2023-04-05 18:03:00 |
| extremo posterior | 2023-04-16 14:18:00 |
| duración exacta | 936.900 s = 260h15min = 10d 20h15min |
| jornadas calendario totalmente sin datos | 10 (2023-04-06 → 2023-04-15 inclusive); ambos extremos (04-05, 04-16) tienen cobertura parcial |
| jornadas de trading afectadas | **no evaluable en S00** — requiere el calendario CME que aún no existe (pertenece a S01) |
| recurrencia estructural | 2 (bucket `>100h`, junto con el gap H25→M25; ningún otro gap del dataset cae en este bucket) |
| interpretación provisional (hipótesis UTC) | "duración muy superior a un cierre de fin de semana típico; sin patrón estructural reconocido en S00" |
| impacto potencial downstream | si resulta pérdida real de datos: ventanas/features/targets que crucen este intervalo en S02-S07 quedarían indefinidas para ese tramo; si resulta evento de mercado/fuente legítimo (ej. mantenimiento extraordinario del proveedor), no requiere tratamiento especial más allá de excluir ventanas que lo crucen |
| **clasificación final** | **`no_resuelto`** (`evidence_level = unconfirmed`) |

### Gap de transición H25 → M25

| campo | valor |
|---|---|
| archivos/contratos | `20_mnq_03_25.Last.txt` (H25) → `21_mnq_06_25.Last.txt` (M25) |
| tipo estructural | `inter_contract` |
| extremo anterior | 2025-03-21 13:30:00 |
| extremo posterior | 2025-04-06 08:42:00 |
| duración exacta | 1.365.120 s = 379h12min = 15d 19h12min |
| jornadas calendario totalmente sin datos | 15 (2025-03-22 → 2025-04-05 inclusive); ambos extremos con cobertura parcial |
| jornadas de trading afectadas | **no evaluable en S00** — requiere calendario CME (S01) |
| recurrencia estructural | 2 (mismo bucket `>100h` que el gap de M23) |
| interpretación provisional (hipótesis UTC) | igual a la anterior: sin patrón estructural reconocido, requiere auditoría adicional |
| impacto potencial downstream | igual naturaleza que el caso anterior; adicionalmente, al ser transición de contrato, cualquier ventana/feature que dependiera de continuidad de rolling entre H25 y M25 debe tratarse con la misma cautela que cualquier roll (ver `01_CURRENT_DECISIONS.md §8`) |
| **clasificación final** | **`no_resuelto`** (`evidence_level = unconfirmed`) |

Ninguno de los dos casos fue rellenado, interpolado, eliminado ni modificado.
Ambos quedan persistidos íntegros en `mnq_raw_v2_gaps.parquet` y resumidos
(sin el detalle de los 4.243 gaps) en manifest/summary.

Los otros 6 gaps del bucket `70min-100h` que superan el umbral de 70h
configurado (`extraordinary_threshold_hours`) corresponden a las 5
transiciones normales de fin de año/roll con feriados encadenados y una
transición de contrato con feriado — todos con `recurrence = 743` (el mismo
bucket que agrupa los ~743 cierres de fin de semana normales) y quedaron
clasificados `provisional_pattern_match`, no `unconfirmed`, precisamente
porque su duración y recurrencia sí son consistentes con el patrón dominante
del dataset. Solo los dos casos de arriba rompen ese patrón.

---

## 7. Resultado de escritura y relectura del Parquet

```
data/01_raw/mnq_raw_v2.parquet
  shape escrito:      (2172640, 6)
  relectura:          equivalente lógicamente (shape, columnas, dtypes, índice y valores) -- verificado dentro de atomic_write_parquet() antes de mover el archivo temporal al nombre final
  SHA-256 final:       dfe1490ac9e9a0c85c4e2ae57c8fbbb90e84a5ee369146941f28e5e047c5beb0
  escritura atómica:   sí (archivo .tmp -> relectura -> Path.replace())

data/01_raw/mnq_raw_v2_gaps.parquet
  shape:               (4243, 12)
  columnas:            gap_type_structural, source_file_left, source_file_right, contract_left,
                        contract_right, previous_timestamp, next_timestamp, duration_seconds,
                        structural_bucket, recurrence, provisional_interpretation_utc_hypothesis,
                        evidence_level
```

No se exigió igualdad byte a byte entre Parquets (la metadata interna puede
variar); se verificó equivalencia lógica de contenido, como acordado.

---

## 8. Comparación documental con S00 anterior

**Etiquetada como documental, no como comparación directa entre artefactos
Parquet** — `data/01_raw/mnq_raw.parquet` original no existe en disco (ver
hallazgo C1 de la auditoría). Se compara contra la salida cacheada de
`S00_raw_data_preparation.ipynb` y `S00_raw_data_preparation_CONTEXT.md`:

| Campo | S00 histórico (documental) | S00 v2 (verificado) | Coincide |
|---|---|---|---|
| Filas | 2.172.640 | 2.172.640 | Sí |
| Columnas | open, high, low, close, volume, contract | open, high, low, close, volume, contract | Sí |
| Rango | 2019-12-23 03:01:00 → 2026-04-17 20:18:00 | idéntico | Sí |
| Contratos | 26 (H20…M26) | 26 (H20…M26) | Sí |
| Duplicados de timestamp | 0 | 0 | Sí |
| Solapamientos entre archivos | 0 (verificado en la auditoría previa por comparación explícita de intervalos) | no re-verificado por un chequeo dedicado en S00 v2 (el módulo no implementa una comparación de intervalos entre archivos, solo duplicados exactos por timestamp/contrato) | No re-probado — pendiente si se quiere una garantía explícita en el módulo |
| Nulos | no reportado explícitamente en v1 | 0 (validado explícitamente) | Mejora |
| Índice tz | tz-naive, "interpretado" como UTC de forma ambigua en el summary v1 (hallazgo C2) | tz-naive, `timezone_assumption=UTC` declarado como no confirmado explícitamente | Corregido |

---

## 9. Problemas pendientes (no resueltos por S00, quedan para S01+)

1. **Zona horaria de origen no confirmada documentalmente** — sigue como
   `inferred_not_confirmed`. Requiere evidencia externa (config de
   exportación de NinjaTrader) que no está disponible en el repositorio.
2. **Dos gaps extraordinarios `no_resuelto`** (§6) — requieren investigación
   adicional fuera del alcance de S00 (contacto con el proveedor de datos,
   o adquisición de una fuente alternativa para ese tramo).
3. **`timestamp_semantics` sin confirmar** — no se sabe si el timestamp
   representa inicio o cierre de la barra de 1 minuto.
4. **`price_type = "Last"` inferido solo del nombre de archivo**, sin
   confirmación documental del proveedor.
5. Los 6 gaps `provisional_pattern_match` en el bucket `70min-100h` son una
   hipótesis estructural razonable, no una confirmación — S01 deberá
   validarlos contra un calendario real de CME.
6. **El módulo no implementa un chequeo dedicado de solapamiento de
   intervalos entre archivos** (distinto de duplicados exactos por
   timestamp/contrato). La auditoría previa lo verificó manualmente (0
   solapamientos), pero esa verificación no quedó automatizada dentro de
   `s00_raw_ingestion.py`. Si se desea la garantía dentro del pipeline, es
   una mejora pendiente menor.

Ninguno de estos puntos bloquea el cierre de S00 según el criterio de
aceptación acordado (documentar y avanzar, no investigar exhaustivamente
ahora), salvo que el usuario decida lo contrario.

---

## 10. Recomendación

**Aprobar S00 v2** como reemplazo funcional de la consolidación cruda,
manteniendo `S00_raw_data_preparation.ipynb` (v1) como evidencia histórica
únicamente.

Justificación: las 35 pruebas pasan, las validaciones de integridad se
ejecutaron sobre el corpus real sin encontrar ninguna fila inválida, el
Parquet se escribió y releyó de forma verificada, el manifiesto es
autoritativo y trazable (código + config + fuentes + git), y la comparación
documental contra S00 v1 coincide en todos los campos verificables. Los
puntos pendientes de §9 son limitaciones de evidencia externa, no defectos
de esta implementación, y ya están explícitamente declarados como no
confirmados en config/manifest/summary — no ocultos.

---

## 11. ¿Puede usarse como entrada para diseñar S01 v2?

**Sí, con las siguientes condiciones explícitas para quien diseñe S01 v2:**

- Partir de `data/01_raw/mnq_raw_v2.parquet` (schema idéntico al histórico:
  `open, high, low, close, volume, contract`, índice tz-naive).
- Tratar `timezone_assumption=UTC` como **hipótesis a confirmar**, no como
  dato confirmado, antes de aplicar `tz_localize`/`tz_convert`.
- Consultar `data/01_raw/mnq_raw_v2_gaps.parquet` al diseñar el calendario y
  la política de jornadas especiales — en particular los dos gaps
  `no_resuelto` y los 6 `provisional_pattern_match`, que S01 sí está en
  posición de reclasificar con un calendario real.
- No asumir `timestamp_semantics` (inicio/cierre de barra) sin decidirlo
  explícitamente, ya que afecta cómo se interpreta `minute_of_day`.

---

## 12. Addendum — Actualización de fuente y saneamiento de código (2026-07-31)

`data/00_source/` fue actualizada por fuera de este pipeline: ahora contiene
**27 archivos** (se agregó `26_mnq_09_26.Last.txt`, contrato U26) y varios
archivos existentes (`13_mnq_06_23`, `20_mnq_03_25`, `21_mnq_06_25`,
`25_mnq_06_26`) se reemplazaron con versiones más completas. Como
consecuencia directa:

```text
Filas totales:        2.172.640 -> 2.329.783
Archivos fuente:       26 -> 27
Rango temporal:        ...2026-04-17 20:18 -> ...2026-07-31 20:10
```

**Los dos gaps extraordinarios documentados como `no_resuelto` (§4.1-bis de
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md`) ya NO existen en los datos
actuales:**

- **S00-05 (gap H25→M25, inter_contract, ~15d19h):** los archivos
  actualizados de H25 y M25 ahora se solapan (H25: 2024-12-12→2025-03-22;
  M25: 2025-03-13→2025-06-22) en vez de tener un vacío entre ambos. Este
  solapamiento es precisamente lo que S01 v2 usa para resolver el rollover
  Z24→H25 y H25→M25 (ver §6 de `S01_v2_report.md`).
- **S00-06 (gap interno M23, intra_file, ~260h):** el archivo M23
  actualizado ya no tiene ningún salto mayor a ~57h dentro de su rango.

`data/01_raw/mnq_raw_v2_gaps.parquet` ya no contiene ningún gap en el
bucket `>100h`; los `70min-100h` restantes (5 casos) son fines de semana
largos ordinarios, no anomalías sin explicar.

**Correcciones de código detectadas y aplicadas durante esta revisión** (no
afectan las filas del dataset, solo la implementación):

- `compute_gaps` había quedado con una firma nueva (`df, infos`) sin que las
  dos pruebas unitarias que la invocaban se actualizaran; corregido.
- Quedaban dos bloques de código muerto (`def_validate_source_filenames_anterior`,
  `compute_gaps_anterior`), versiones previas de funciones guardadas como
  strings sin usar; eliminados.
- Se agregó una prueba unitaria explícita
  (`test_overlapping_contracts_same_timestamp_produce_no_gap`) que fija
  como contrato de regresión que dos contratos con timestamps compartidos
  durante un rollover NO generan una fila en el catálogo de gaps — distinto
  de un duplicado real (`timestamp`+`contract` idénticos), que sigue
  deteniendo la ingestión.

**Pendientes de S00 sin cambios:** confirmación documental de zona horaria
y de `timestamp_semantics` siguen sin resolverse (no hay evidencia nueva
del proveedor). El chequeo automatizado de solapamiento entre archivos
sigue siendo una mejora menor pendiente.

Reejecutado: 26/26 pruebas unitarias + 10/10 de integración de S00 pasan
contra el corpus actualizado.
