# S03 — Threshold Calibration

## 1. Identificación

- **Notebook:** `S03_threshold_calibration.ipynb`
- **Etapa:** Stage 03
- **Función:** calcular, diagnosticar, seleccionar y guardar thresholds operativos para construir los targets del Stage 04.
- **Estado:** ejecutada y utilizada por etapas posteriores. Sus resultados por régimen están condicionados por la clasificación de regímenes heredada de S01.

## 2. Posición dentro del pipeline

```text
S00 → S01 → S02 → S03 → Stage 04
```

Entrada principal:

```text
data/02_mnq_intraday/mnq_intraday.parquet
```

Dataset cargado:

```text
Shape: 1.024.062 × 9
Periodo: 2020-01-02 04:30 → 2026-04-17 16:00
Zona horaria: America/New_York
Días: 1.482
Índice temporal validado: ordenado, único y continuo dentro de cada día
```

S03 no construye targets. Genera los thresholds que posteriormente usa Stage 04.

## 3. Objetivo metodológico

Determinar qué magnitud de movimiento futuro debe considerarse significativa según:

```text
horizonte
percentil
régimen intradiario
```

La calibración se realiza exclusivamente con:

```text
Development: 2020–2024
Final test reservado: 2025–2026
```

Distribución:

```text
development_2020_2024: 818.835 filas, 79,96 %
final_test_2025_2026: 205.227 filas, 20,04 %
```

El periodo 2025–2026 no participa en la selección de thresholds.

## 4. Excursiones futuras

Horizontes:

```python
THRESHOLD_HORIZONS = [30, 60, 90]
```

Para cada instante `t` y horizonte `H`, la ventana es:

```text
t+1, ..., t+H
```

La barra actual queda excluida y la ventana no cruza días.

La notebook calcula sobre precios de cierre:

```text
future_close_max_Hm
future_close_min_Hm
up_excursion_pts_Hm   = future_close_max_Hm - close(t)
down_excursion_pts_Hm = close(t) - future_close_min_Hm
max_excursion_pts_Hm  = max(up_excursion, down_excursion)
```

Resultado:

```text
df_threshold_excursions: 1.024.062 × 24
```

Filas válidas en todo el dataset:

| Horizonte | Válidas | Inválidas | Ratio válido |
|---:|---:|---:|---:|
| 30 | 979.602 | 44.460 | 95,66 % |
| 60 | 935.142 | 88.920 | 91,32 % |
| 90 | 890.682 | 133.380 | 86,98 % |

Las filas inválidas se concentran al final de cada jornada por falta de barras futuras.

## 5. Percentiles estudiados

```python
THRESHOLD_PERCENTILES_MAIN = [40, 50, 60]
THRESHOLD_PERCENTILES_ALL = [25, 40, 50, 60, 75, 90, 95]
```

Interpretación operativa:

```text
p40 → flexible
p50 → base
p60 → exigente
```

Se calculan tres tipos:

```text
threshold_up_pts
threshold_down_pts
threshold_common_pts
```

El candidato inicial para Stage 04 es:

```text
threshold_common_pts
```

`threshold_common_pts` es el percentil de `max_excursion_pts`, no el máximo entre los percentiles up y down.

## 6. Niveles de análisis

Se calcularon thresholds:

```text
globales
por régimen
por régimen y año
por régimen y trimestre
por régimen y contrato
por régimen y familia de contrato
```

Los niveles muy segmentados se utilizaron para diagnóstico de estabilidad, no como definición operativa principal.

## 7. Thresholds globales principales

`threshold_common_pts` calculado con 2020–2024:

| Horizonte | p40 | p50 | p60 |
|---:|---:|---:|---:|
| 30 | 21,00 | 25,50 | 31,00 |
| 60 | 31,75 | 38,50 | 46,50 |
| 90 | 40,50 | 49,00 | 59,50 |

Estos valores se conservan como benchmark, no como metodología principal.

## 8. Metodología seleccionada

La metodología principal es:

```text
regime_pooled_2020_2024
```

Agrupación:

```text
horizon + percentile + regime_id
```

Se seleccionó porque:

- los movimientos difieren fuertemente entre regímenes;
- un threshold global es demasiado alto para Overnight;
- un threshold global es demasiado bajo para Opening y Pre-market;
- la segmentación por contrato o trimestre introduce demasiada inestabilidad;
- el pooled por régimen se mantiene razonablemente cerca de la mediana anual.

Control robusto:

```text
regime_median_year_2020_2024
```

Benchmark:

```text
global_pooled_2020_2024
```

## 9. Thresholds principales por régimen

### `threshold_common_pts`, método pooled

| Horizonte | Percentil | Overnight | Pre-market | Opening | Regular | Closing |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | p40 | 14,00 | 25,25 | 42,75 | 25,50 | 30,00 |
| 30 | p50 | 16,50 | 31,00 | 50,00 | 30,00 | 35,50 |
| 30 | p60 | 19,25 | 38,25 | 58,50 | 35,50 | 41,75 |
| 60 | p40 | 21,00 | 54,00 | 58,50 | 37,25 | — |
| 60 | p50 | 24,75 | 63,25 | 68,50 | 44,00 | — |
| 60 | p60 | 29,25 | 74,00 | 79,75 | 51,50 | — |
| 90 | p40 | 28,00 | 73,75 | 68,75 | 46,25 | — |
| 90 | p50 | 33,00 | 85,50 | 80,25 | 54,25 | — |
| 90 | p60 | 39,25 | 99,00 | 93,50 | 63,50 | — |

La combinación que luego adquirió mayor prioridad en el proyecto fue:

```text
p50 + h60 + threshold_common_pts + régimen
```

## 10. Robustez temporal

Se comparó el método pooled con la mediana de thresholds calculados por año dentro de cada régimen.

Resultado declarado:

```text
Diferencia media aproximada: 6,8 %–10,4 %
Diferencia máxima: inferior a aproximadamente 14 %
Ratio pooled / mediana anual: aproximadamente 1,03–1,12
```

El pooled tiende a ser ligeramente más exigente, pero no se consideró severamente distorsionado por años extremos.

Se observó que 2022 presenta thresholds altos en numerosos contextos.

## 11. Artefactos generados

Intermedios:

```text
data/03_mnq_thresholds/mnq_threshold_excursions.parquet
data/03_mnq_thresholds/mnq_threshold_excursions_invalid_summary.parquet
data/03_mnq_thresholds/mnq_threshold_excursions_summary.json
```

Finales:

```text
mnq_thresholds_global_benchmark.parquet
mnq_thresholds_regime_primary.parquet
mnq_thresholds_regime_robust_check.parquet
mnq_thresholds_final_candidates.parquet
mnq_thresholds_final_candidates.csv
mnq_thresholds_final_summary.json
```

Tabla consolidada:

```text
df_thresholds_final_candidates: 87 × 27
```

Composición:

```text
global benchmark: 9 filas
regime pooled primary: 39 filas
regime median-year robustness check: 39 filas
```

Validación final:

```text
IDs duplicados: 0
threshold_common faltantes: 0
threshold_common no positivos: 0
threshold_up faltantes: 0
threshold_down faltantes: 0
```

## 12. Problemas críticos

### 12.1. Regímenes heredados de S01

S03 usa directamente `regime_id` de `mnq_intraday`.

La clasificación realmente presente era:

```text
0 Overnight: 04:30–08:29 y barra 16:00
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–15:29
4 Closing: 15:30–15:59
```

La convención aprobada posteriormente es:

```text
0 Overnight: 04:30–08:29
1 Pre-market: 08:30–09:29
2 Opening: 09:30–10:29
3 Regular: 10:30–14:59
4 Closing: 15:00–16:00
```

Consecuencias:

- los thresholds de Overnight incluyen erróneamente las barras de 16:00;
- Regular incorpora 30 minutos que posteriormente pertenecen a Closing;
- Closing contiene solo 30 barras por día;
- Closing tiene 1.185 observaciones válidas para H30 y ninguna para H60/H90;
- las 39 combinaciones por régimen, en vez de 45, se explican por la ausencia de Closing en H60/H90;
- todos los targets posteriores construidos con estos thresholds heredan esta definición.

### 12.2. Excursiones basadas únicamente en `close`

La calibración utiliza máximos y mínimos de cierres futuros, no `high` y `low`.

Esto es válido solo si los targets posteriores miden el mismo fenómeno. Si TP/SL se evalúa mediante máximos y mínimos intrabar, existiría una diferencia entre:

```text
fenómeno usado para calibrar el threshold
fenómeno usado para determinar el resultado operativo
```

Stage 04 debe confirmar explícitamente esta coherencia.

### 12.3. Excursiones direccionales no truncadas

`up_excursion` y `down_excursion` no se limitan inferiormente a cero. Si todos los cierres futuros se encuentran de un solo lado del cierre actual, la excursión contraria puede ser negativa.

No afectó la validación de `threshold_common_pts`, pero debe documentarse si se utilizan thresholds direccionales.

### 12.4. Dependencias heredadas

S03 hereda de S01:

```text
calendario NASDAQ aplicado a MNQ
eliminación de jornadas con menos de 691 barras
clasificación incorrecta de regímenes
```

## 13. Decisiones vigentes y reemplazadas

Continúan vigentes:

```text
horizontes 30/60/90
percentiles p40/p50/p60
periodo de desarrollo 2020–2024
reserva de 2025–2026
ventanas futuras sin cruzar días
thresholds por régimen como metodología principal
threshold global como benchmark
p50 como escenario base
threshold_common_pts como opción inicial
```

Análisis no utilizados como metodología principal:

```text
regime_year
regime_year_quarter
regime_contract
regime_contract_family
```

## 14. Relación con el estado actual

Stage 04 utilizó estos thresholds para construir las familias de targets:

```text
DIR
BAR
OPC
```

El target actualmente priorizado es:

```text
opc_p50_h60_tp15_sl10
```

Por lo tanto, cualquier corrección de regímenes o de la definición de excursiones podría modificar:

```text
thresholds
targets
features seleccionadas
datasets predictivos
resultados de Stage 06
diseño y entrenamiento de Stage 07
```

## 15. Estado y acciones pendientes

**Aprobado conceptualmente:**

- separación 2020–2024 / 2025–2026;
- causalidad de ventanas;
- comparación global contra régimen;
- selección de p40/p50/p60;
- pooled por régimen como método principal;
- benchmark global y control por mediana anual.

**Pendiente antes de considerar el pipeline definitivo:**

1. corregir o ratificar la definición de regímenes;
2. recalcular thresholds si se corrigen los regímenes;
3. verificar en Stage 04 si TP/SL usa `close` o `high/low`;
4. decidir si las excursiones direccionales deben truncarse en cero;
5. evaluar el impacto de cualquier corrección sobre todos los artefactos posteriores;
6. conservar los Parquet pesados fuera del repositorio y entregar a Claude solo esta ficha y resultados puntuales.
