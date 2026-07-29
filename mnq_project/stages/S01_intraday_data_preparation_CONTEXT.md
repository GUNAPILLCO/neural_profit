# S01 — Intraday Data Preparation

## 1. Identificación

- **Notebook:** `S01_intraday_data_preparation.ipynb`
- **Etapa:** Stage 01
- **Función:** transformar `mnq_raw` en el dataset intradiario limpio utilizado por las etapas posteriores.
- **Estado:** ejecutada, pero requiere correcciones para garantizar reproducibilidad y coherencia metodológica.

## 2. Dependencia de entrada

Proviene de `S00_raw_data_preparation.ipynb`.

```text
data/01_raw/mnq_raw.parquet
data/01_raw/mnq_raw_summary.json
```

Dataset inicial cargado:

```text
Shape: 2.172.640 × 6
Columnas: open, high, low, close, volume, contract
Periodo: 2019-12-23 03:01 → 2026-04-17 20:18
Índice original: tz-naive, interpretado como UTC
```

## 3. Procedimiento implementado

1. Carga `mnq_raw.parquet` y su resumen JSON.
2. Ordena el índice y revisa una jornada de ejemplo.
3. Infiere que los timestamps representan UTC mediante el patrón intradiario de volumen.
4. Verifica:
   - filas completamente duplicadas;
   - timestamps duplicados;
   - conflictos de cierre en timestamps repetidos.
5. Filtra fechas mediante el calendario `NASDAQ`.
6. Localiza el índice como UTC y lo convierte a `America/New_York`.
7. Conserva el intervalo inclusivo `04:30–16:00`.
8. Calcula la cantidad modal de registros por día: 691.
9. Elimina jornadas con menos de 691 registros o gaps distintos de un minuto.
10. Verifica continuidad minuto a minuto y ausencia de NaN.
11. Agrega:
    - `date`;
    - `minute_of_day`;
    - `regime_id`.
12. Guarda el dataset y un resumen JSON.

## 4. Resultado final declarado

```text
Nombre: mnq_intraday
Shape: 1.024.062 × 9
Periodo: 2020-01-02 04:30 → 2026-04-17 16:00
Zona horaria: America/New_York
Días completos: 1.482
Registros por día: 691
minute_of_day: 270–960
NaN: 0
Gaps intradiarios: 0
```

La igualdad:

```text
1.482 × 691 = 1.024.062
```

confirma que el dataset final contiene exactamente 691 barras por jornada.

## 5. Salidas

```text
data/02_mnq_intraday/mnq_intraday.parquet
data/02_mnq_intraday/mnq_intraday_summary.json
```

Columnas finales esperadas:

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

## 6. Decisiones metodológicas vigentes

- Zona horaria operativa: `America/New_York`.
- Ventana de análisis: `04:30–16:00`, con ambos extremos incluidos.
- Frecuencia requerida: un minuto.
- Se conservan únicamente jornadas con cobertura completa de 691 barras.
- Las secuencias y horizontes posteriores deben respetar el límite de cada jornada.
- `minute_of_day` se calcula en horario de Nueva York.
- El dataset generado constituye la base consolidada usada desde Stage_02.

## 7. Problemas críticos detectados

### 7.1. Calendario incorrecto para el instrumento

La notebook utiliza:

```python
mcal.get_calendar("NASDAQ")
```

MNQ es un futuro negociado en CME Globex, no una acción negociada en NASDAQ. El calendario correcto debe corresponder a futuros de índices de CME o definirse explícitamente.

Esto puede afectar feriados, cierres anticipados y jornadas especiales.

### 7.2. Orden temporal del filtrado

Los días de mercado se filtran antes de localizar UTC y convertir a Nueva York. Aunque la ventana final `04:30–16:00` cae normalmente en la misma fecha UTC y local, metodológicamente es más seguro:

```text
localizar UTC → convertir a New York → asignar fecha operativa → aplicar calendario
```

### 7.3. Eliminación de jornadas incompletas

Se eliminaron 80 de 1.562 jornadas después del filtrado horario.

No todas tienen que ser errores de datos: algunas pueden corresponder a cierres anticipados legítimos. La eliminación basada en 691 registros debe distinguir entre:

- gaps o archivos incompletos;
- sesiones especiales válidas;
- cierres anticipados programados.

Esta selección no introduce directamente el target futuro, pero sí condiciona el dataset a días conocidos como completos. La política para días especiales debe quedar definida antes del uso en producción.

### 7.4. Regímenes inconsistentes

La documentación de la notebook contiene varias definiciones distintas:

```text
Regular hasta 15:30 / Closing desde 15:30
Regular hasta 15:00 / Closing desde 15:00
Opening hasta 10:30 o hasta 11:00
```

El código implementado usa:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–15:29
4 Closing: 15:30–15:59
```

Además, la barra de `16:00` no coincide con ninguna condición y recibe el valor por defecto:

```text
regime_id = 0
```

Esto es incorrecto.

La convención aprobada posteriormente en el proyecto es:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–14:59
4 Closing: 15:00–16:00
```

S01 debe alinearse con esta convención.

### 7.5. Artefacto potencialmente desactualizado

El bloque de guardado funciona así:

```text
si el Parquet existe → lo carga
si no existe → lo guarda
```

Por lo tanto, al modificar el código y volver a ejecutar la notebook, el archivo existente reemplaza el DataFrame recién calculado. El resumen final puede corresponder a una versión anterior y no al código actual.

La notebook debe reconstruir y sobrescribir explícitamente el artefacto, o utilizar una opción controlada como `FORCE_REBUILD`.

### 7.6. Zona horaria inferida, no verificada documentalmente

UTC se infiere observando el volumen de una jornada. Es una evidencia razonable, pero debe confirmarse con:

- configuración de exportación de NinjaTrader;
- metadatos de la fuente;
- documentación del proveedor.

### 7.7. Validaciones declaradas pero no ejecutadas

La sección final afirma haber validado:

```text
high ≥ open/close ≥ low
volumen no negativo
ausencia de valores aberrantes
```

Estas comprobaciones no aparecen implementadas.

También faltan verificaciones explícitas de:

- infinitos;
- tipos de datos;
- precios no positivos;
- cobertura exacta de límites horarios;
- distribución y exhaustividad de `regime_id`;
- coherencia entre `date` y el índice;
- duplicados después de todas las transformaciones.

## 8. Inconsistencias documentales menores

La notebook menciona alternativamente:

```text
08:30–16:00
06:30–16:00
04:30–16:00
```

El código y el dataset final usan `04:30–16:00`.

La sección “Alineación con libro ML” declara aproximadamente 744.000 registros, pero el resultado real es 1.024.062.

## 9. Dependencias posteriores

`mnq_intraday.parquet` alimenta:

- análisis estadístico intradiario;
- cálculo de excursiones y thresholds;
- construcción de targets;
- feature engineering;
- validación walk-forward;
- generación de secuencias del Stage_07.

Cualquier cambio en horario, calendario, días eliminados o regímenes puede modificar todas las etapas posteriores. No debe regenerarse el archivo sin evaluar su impacto global.

## 10. Estado y acciones pendientes

**Aprobado conceptualmente:**

- conversión a `America/New_York`;
- ventana `04:30–16:00`;
- frecuencia de un minuto;
- dataset de 1.482 días y 1.024.062 filas;
- ausencia de NaN y gaps en el artefacto utilizado posteriormente.

**Pendiente:**

1. Sustituir el calendario NASDAQ por el calendario aplicable a MNQ/CME.
2. Auditar las 80 jornadas eliminadas.
3. Corregir y validar la clasificación de regímenes.
4. Evitar la recarga silenciosa de un Parquet antiguo.
5. Confirmar documentalmente que la fuente está en UTC.
6. Implementar las validaciones OHLCV declaradas.
7. Corregir el Markdown para que coincida con el código y el pipeline vigente.
8. No regenerar etapas posteriores hasta decidir si estos cambios alteran el dataset histórico oficial.

Todo lo anterior en este documento describe **exclusivamente la notebook v1**
(`S01_intraday_data_preparation.ipynb`), que no fue modificada y se
conserva como evidencia histórica. Sus resultados (regímenes con límites
incorrectos, calendario NASDAQ, 80 días eliminados sin clasificar) **no
deben presentarse como vigentes.** El cierre real de la etapa se realizó
mediante una implementación nueva (v2) — ver §11.

---

## 11. S01 v2 — Cierre aprobado (actualización)

**S01 v2 fue aprobado formalmente.** Es la implementación vigente.

```text
Implementación:    src/data/s01_intraday_preparation.py
Config:            config/intraday_config.yaml
Notebook vigente:  notebooks/S01_intraday_data_preparation_v2.ipynb
Artefacto oficial: data/02_intraday/mnq_intraday_v2.parquet
Auditoría de días: data/02_intraday/trading_day_audit_v2.parquet
Distribución:      data/02_intraday/regime_distribution_v2.parquet
Manifiesto:        data/02_intraday/mnq_intraday_v2_manifest.json (autoritativo)
Validación tz:     data/02_intraday/tz_validation_v2.json
Reporte:           reports/stage_reports/S01_v2_report.md
Pruebas:           42/42 aprobadas (26 unitarias + 16 de integración)
Filas:             1.087.777 (subconjunto full_coverage: 1.482 × 691 = 1.024.062, idéntico a v1)
```

### Resolución de los 8 pendientes de §10

| # | Pendiente (v1) | Resolución en S01 v2 |
|---|---|---|
| 1 | Calendario NASDAQ → CME/MNQ | Resuelto: calendario híbrido `CME_Equity` + datos observados como evidencia principal; ningún día se excluye silenciosamente |
| 2 | Auditar las 80 jornadas eliminadas | Resuelto de forma más amplia: las 2.309 fechas del rango quedan clasificadas (no solo las 80 antiguas), con motivo explícito por fecha |
| 3 | Corregir clasificación de regímenes | Resuelto: límites vigentes (`Regular` 10:30-14:59, `Closing` 15:00-16:00), sin ruta default, verificado con prueba de regresión sobre los 9 puntos límite; `regime_id=0` renombrado `Overnight → Early_Premarket` |
| 4 | Recarga silenciosa de Parquet | Resuelto: manifest autoritativo con hash de fuente/módulo/config + `FORCE_REBUILD`, mismo patrón que S00 v2 |
| 5 | Confirmar documentalmente UTC | **No resuelto documentalmente** (sigue sin config del proveedor), pero sustancialmente reforzado: UTC seleccionado por comparación empírica programática de 3 hipótesis (score 2.0 vs 1087.1/1105.1), declarado `timezone_provider_confirmation: false` |
| 6 | Validaciones OHLCV declaradas | Fuera de alcance de S01 (ya garantizado por S00 v2); S01 v2 valida lo que le compete: régimen, ventana, consecutividad, DST |
| 7 | Markdown no coincide con el código | Resuelto: notebook v2 delgada, sin bloques de texto que describan operaciones no implementadas |
| 8 | No regenerar etapas posteriores sin decidir impacto | Aplicado: S02+ no fue tocado; el impacto del cambio de calendario/regímenes queda documentado para cuando se aborde S02 |

### Pendiente que sigue abierto (no bloqueante)

```text
244 jornadas partial_undetermined + 11 no_data_undetermined (causa sin determinar)
Patrón recurrente 16:20-16:30 (2019-2021): fuera de la ventana primaria, no
  afecta mnq_intraday_v2.parquet
Discrepancia CME_Equity vs "CME Globex Equity" (1 día, 2025-01-09): no revalidada
Confirmación documental de zona horaria de origen: sigue pendiente
test_s00_integration.py::test_never_writes_to_productive_raw_dir: falla
  preexistente de S00 (asume data/01_raw/ vacío), no relacionada con S01 v2
```

Detalle completo en `reports/stage_reports/S01_v2_report.md`,
`01_CURRENT_DECISIONS.md §32` y
`02_KNOWN_ISSUES_AND_INVALIDATED_RESULTS.md §4.2-bis`.
