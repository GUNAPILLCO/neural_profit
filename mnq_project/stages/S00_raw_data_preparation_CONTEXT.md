# S00 — Raw Data Preparation

## 1. Identificación

- **Notebook:** `S00_raw_data_preparation.ipynb`
- **Etapa:** Stage 00
- **Función real:** ingestión y consolidación inicial de los históricos crudos del MNQ.
- **Estado:** ejecutada; requiere corregir documentación y algunas inconsistencias antes de considerarla cerrada.

## 2. Propósito dentro del pipeline

Construir el dataset crudo consolidado que alimenta las etapas posteriores. Esta notebook no genera todavía el dataset intradiario limpio y filtrado; únicamente carga, concatena, ordena y documenta los archivos históricos originales.

## 3. Fuente de datos

- **Instrumento:** Micro E-mini Nasdaq-100 Futures (`MNQ`).
- **Frecuencia:** barras de 1 minuto.
- **Formato:** archivos `.txt` separados por `;`.
- **Columnas originales:** `datetime`, `open`, `high`, `low`, `close`, `volume`.
- **Origen declarado:** NinjaTrader.
- **Zona horaria declarada:** UTC.
- **Ruta de entrada:** `data/00_source/`.
- **Cantidad detectada:** 26 archivos trimestrales, desde `MNQH20` hasta `MNQM26`.

## 4. Procedimiento implementado

1. Localiza la raíz del proyecto `neural_profit`.
2. Revisa el rango temporal y la cantidad de filas de cada archivo fuente.
3. Extrae el contrato trimestral desde el nombre del archivo:
   - marzo → `H`
   - junio → `M`
   - septiembre → `U`
   - diciembre → `Z`
4. Comprueba si existen superposiciones temporales entre archivos.
5. Lee todos los `.txt`, asigna tipos a OHLCV y convierte `datetime`.
6. Agrega la columna `contract`.
7. Concatena todos los contratos y ordena cronológicamente el índice.
8. Guarda o reutiliza el dataset consolidado en Parquet.
9. Verifica timestamps duplicados.
10. Genera un resumen descriptivo en JSON.

## 5. Entradas y salidas

### Entrada

```text
data/00_source/*.txt
```

### Salidas

```text
data/01_raw/mnq_raw.parquet
data/01_raw/mnq_raw_summary.json
```

## 6. Resultado principal

Dataset generado:

```text
Nombre: mnq_raw
Shape: 2.172.640 filas × 6 columnas
Columnas: open, high, low, close, volume, contract
Inicio: 2019-12-23 03:01:00
Fin: 2026-04-17 20:18:00
Días con datos: 2.005
Cobertura horaria observada: 00:00–23:59
Timestamps duplicados: 0
Superposiciones entre archivos: 0
```

El dataset continúa representando datos crudos de cobertura prácticamente completa y no una sesión intradiaria ya filtrada.

## 7. Decisiones metodológicas vigentes

- Los contratos trimestrales se identifican a partir del nombre del archivo.
- Los históricos se consolidan en orden cronológico.
- El formato persistente principal es Parquet.
- `datetime` se utiliza como índice.
- No se eliminan registros en esta etapa.
- No se aplican aquí filtros por sesión, régimen, días hábiles o calendario bursátil.
- La limpieza y transformación temporal deben ejecutarse en notebooks posteriores.

## 8. Inconsistencias y advertencias

### 8.1. El resumen inicial no coincide con el código

El Markdown de la notebook afirma que se realizan:

- eliminación de fines de semana y feriados;
- conversión UTC → US/Eastern;
- filtrado de 08:30–16:00;
- eliminación de días incompletos;
- verificación de continuidad minuto a minuto;
- generación de un dataset final limpio.

Ninguna de esas operaciones está implementada realmente en esta notebook. Su función efectiva es solo generar `mnq_raw`.

### 8.2. Zona horaria

El índice cargado es `tz-naive`. La función de resumen lo localiza temporalmente como UTC para informar estadísticas, pero no modifica el índice de `df_mnq_raw`.

Por lo tanto:

```text
df_mnq_raw.index.tz == None
```

Las etapas posteriores deben localizar explícitamente el índice en UTC antes de convertirlo a `America/New_York`.

### 8.3. Nomenclatura inconsistente de contratos

La función de inspección devuelve códigos como:

```text
MNQH20, MNQM20, ...
```

La función que construye el dataset guarda:

```text
H20, M20, ...
```

Debe definirse una única convención. El pipeline posterior utiliza actualmente la forma abreviada `H20`, `M20`, `U20`, `Z20`.

### 8.4. Conteo diario mal interpretado

La notebook marca como sospechosos los días con más de 500 filas, basándose en una expectativa de 390–450 barras. Sin embargo, `mnq_raw` contiene datos de casi todo el día y puede alcanzar aproximadamente 1.380 barras.

Este chequeo no detecta duplicación válida en el dataset crudo. La prueba confiable es la duplicación exacta del timestamp, cuyo resultado fue cero.

### 8.5. Archivo `MNQM25`

El archivo correspondiente a junio de 2025 comienza el `2025-04-06`, mientras que el contrato anterior termina el `2025-03-21`. Existe un periodo sin cobertura entre ambos archivos que debe quedar documentado y revisarse en la etapa de calidad temporal.

## 9. Dependencias posteriores

Las notebooks siguientes deben tomar como entrada:

```text
data/01_raw/mnq_raw.parquet
```

y encargarse de:

- localizar correctamente la zona horaria UTC;
- convertir a `America/New_York`;
- definir el rango horario operativo;
- eliminar días o registros no válidos;
- detectar gaps;
- generar variables temporales y regímenes;
- producir el dataset intradiario validado.

## 10. Estado y acciones pendientes

**Resultado aprobado:** consolidación de 26 archivos, orden temporal, ausencia de solapamientos y ausencia de timestamps duplicados.

**Pendiente antes de cerrar S00:**

1. Corregir el resumen Markdown para describir solamente lo ejecutado.
2. Unificar la nomenclatura de `contract`.
3. Aclarar que el Parquet conserva un índice `tz-naive` interpretado como UTC.
4. Reemplazar o eliminar el chequeo de días con más de 500 registros.
5. Documentar el gap previo a `MNQM25`.
6. Confirmar valores faltantes y consistencia OHLCV mediante validaciones explícitas.

Todo lo anterior en este documento describe **exclusivamente la notebook v1**
(`S00_raw_data_preparation.ipynb`), que no fue modificada y se conserva como
evidencia histórica. El cierre real de la etapa se realizó mediante una
implementación nueva (v2, no una corrección de la notebook v1) — ver §11.

---

## 11. S00 v2 — Cierre aprobado (actualización)

**S00 v2 fue aprobado formalmente.** Es la implementación vigente; la
notebook v1 y este documento hasta el §10 quedan como registro histórico de
lo que motivó la reconstrucción.

```text
Implementación:    src/data/s00_raw_ingestion.py
Notebook vigente:  notebooks/S00_raw_data_preparation_v2.ipynb
Config:            config/data_config.yaml
Artefacto crudo:   data/01_raw/mnq_raw_v2.parquet
Manifiesto:        data/01_raw/mnq_raw_v2_manifest.json (autoritativo)
Summary:           data/01_raw/mnq_raw_v2_summary.json
Gaps:              data/01_raw/mnq_raw_v2_gaps.parquet
Reporte:           reports/stage_reports/S00_v2_report.md
Pruebas:           35/35 aprobadas (25 unitarias + 10 de integración)
Filas:             2.172.640
Filas rechazadas:  0
```

### Resolución de los 6 pendientes de §10

| # | Pendiente (v1) | Resolución en S00 v2 |
|---|---|---|
| 1 | Markdown no coincide con el código | Resuelto: la notebook v2 es delgada, sin bloques de texto que describan operaciones no implementadas |
| 2 | Nomenclatura de `contract` inconsistente | Resuelto: función única de extracción; dataset conserva formato corto (`H20`), `instrument`/`contract_full` quedan en el manifiesto, no en el dataset |
| 3 | Aclarar tz-naive/UTC | Resuelto documentalmente: `timezone_assumption: "UTC"`, `timezone_evidence: "inferred_not_confirmed"` explícitos en config/manifest/summary; el índice sigue tz-naive, sin conversión |
| 4 | Chequeo de >500 filas/día | Resuelto: no existe en S00 v2; se reemplazó por validaciones explícitas de esquema, OHLC, nulos, infinitos y duplicados |
| 5 | Gap previo a `MNQM25` | Cuantificado y clasificado (`no_resuelto`, `evidence_level=unconfirmed`) en `mnq_raw_v2_gaps.parquet`; la causa raíz **sigue sin determinarse** |
| 6 | Confirmar nulos/OHLCV explícitamente | Resuelto: validado sobre las 2.172.640 filas reales, 0 rechazos, con pruebas de regresión para cada invariante |

### Hallazgo nuevo no cubierto por los 6 pendientes de v1

Un segundo gap extraordinario, no documentado antes de esta reconstrucción:
gap interno de ~260h15min dentro de `13_mnq_06_23.Last.txt` (contrato M23),
entre 2023-04-05 18:03:00 y 2023-04-16 14:18:00. Clasificado `no_resuelto`
en `mnq_raw_v2_gaps.parquet`, igual que el gap de `MNQM25`.

### Pendiente que sigue abierto (no bloqueante)

```text
Confirmación documental de zona horaria de origen (proveedor/exportación)
Semántica del timestamp (inicio vs. cierre de barra) — no confirmada
Gap interno MNQM23 — no_resuelto
Gap de transición H25→M25 — no_resuelto
Chequeo automatizado de solapamiento de intervalos entre archivos —
  mejora menor pendiente (verificado manualmente en la auditoría, no
  automatizado dentro del módulo)
```

Detalle completo en `reports/stage_reports/S00_v2_report.md`,
`01_CURRENT_DECISIONS.md §31` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.1-bis`.

### Addendum (2026-07-31)

`data/00_source/` se actualizó por fuera de este pipeline (27 archivos en
vez de 26; se agregó `26_mnq_09_26.Last.txt` y se reemplazaron 4 archivos
existentes con versiones más completas). Efecto directo:

```text
Filas: 2.172.640 -> 2.329.783
Rango: ...2026-04-17 20:18 -> ...2026-07-31 20:10
```

Los dos gaps `no_resuelto` de esta tabla (gap previo a MNQM25 y el gap
interno de MNQM23) **ya no existen en los datos actuales** — quedan
`RESUELTO` por datos reales, no por decisión de alcance. Ver
`reports/stage_reports/S00_v2_report.md §12` para el detalle completo,
incluidas dos correcciones menores de código encontradas durante esta
revisión (firma desactualizada de `compute_gaps` en dos pruebas, y dos
bloques de código muerto eliminados).
