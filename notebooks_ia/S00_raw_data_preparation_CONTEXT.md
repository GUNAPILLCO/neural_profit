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
