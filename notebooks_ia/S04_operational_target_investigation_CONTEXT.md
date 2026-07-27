# S04 — Operational Target Investigation

## 1. Identificación

- **Notebook:** `S04_operational_target_investigation.ipynb`
- **Etapa:** Stage 04
- **Función:** construir, validar, analizar y preseleccionar targets `DIR`, `BAR` y `OPC`.
- **Estado:** ejecutada y utilizada por stages posteriores.
- **Nota de vigencia:** la notebook eligió preliminarmente `opc_p50_h30_tp15_sl10`, pero esta decisión fue reemplazada posteriormente. El target principal vigente para Stage_07 es `opc_p50_h60_tp15_sl10`.

## 2. Posición dentro del pipeline

```text
S00 → S01 → S02 → S03 → S04 → Stage_05
```

Entradas principales:

```text
data/02_mnq_intraday/mnq_intraday.parquet
data/03_mnq_thresholds/mnq_thresholds_regime_primary.parquet
```

Dataset base:

```text
1.024.062 filas
9 columnas
2020-01-02 04:30 → 2026-04-17 16:00
America/New_York
1.482 jornadas
```

Thresholds utilizados:

```text
Método: regime_pooled_2020_2024
Tipo: threshold_common_pts
Horizontes: 30, 60, 90
Percentiles: 40, 50, 60
Condicionamiento: horizon + percentile + regime_id
Filas de thresholds: 39
```

## 3. Separación temporal

```text
Development: 2020–2024
Final test declarado: 2025–2026
```

Los thresholds permanecen congelados al etiquetar 2025–2026.

Advertencia: aunque 2025–2026 no recalibra los thresholds, sus resultados son analizados y considerados en la preselección descriptiva. Para un holdout final completamente intacto, ese periodo no debería influir tampoco en la elección del target.

---

# 4. Target DIR

## 4.1. Definición

Para cada instante `t`, horizonte `H` y threshold por régimen:

```text
future window = t+1, ..., t+H
up_excursion   = max(close futuro) - close(t)
down_excursion = close(t) - min(close futuro)
```

Clases:

```text
 1 = LONG
-1 = SHORT
 0 = NO_SIGNAL
NaN = ventana futura incompleta
```

Reglas:

```text
LONG  si up >= threshold y up > down
SHORT si down >= threshold y down > up
0     en los demás casos válidos
```

Las ventanas se construyen por `date` y no cruzan jornadas.

## 4.2. Targets construidos

```text
dir_p40_h30  dir_p50_h30  dir_p60_h30
dir_p40_h60  dir_p50_h60  dir_p60_h60
dir_p40_h90  dir_p50_h90  dir_p60_h90
```

Resultado guardado:

```text
mnq_targets_dir.parquet
Shape: 1.024.062 × 31
```

Filas con ventana válida:

```text
H30: 979.602
H60: 935.142
H90: 890.682
```

Frecuencia de señales:

| Target | LONG | SHORT | NO_SIGNAL | Señal total |
|---|---:|---:|---:|---:|
| `dir_p50_h30` | 26,53 % | 27,57 % | 45,90 % | 54,10 % |
| `dir_p50_h60` | 26,34 % | 27,67 % | 46,00 % | 54,00 % |
| `dir_p50_h90` | 26,34 % | 27,85 % | 45,82 % | 54,18 % |

`p40` es más activo; `p60`, más selectivo. Existe una leve inclinación hacia SHORT.

## 4.3. DIR preseleccionados en S04

```text
Principales:
dir_p50_h30
dir_p50_h60
dir_p50_h90

Comparativos:
dir_p40_h60
dir_p60_h60
```

---

# 5. Target BAR

## 5.1. Definición

BAR se construye solo cuando DIR define una dirección:

```text
DIR = 1  → operación LONG
DIR = -1 → operación SHORT
DIR = 0  → BAR = NaN
```

Las barreras son múltiplos del threshold correspondiente:

```text
TP distance = tp_mult × threshold
SL distance = sl_mult × threshold
```

Clases:

```text
 1 = TP alcanzado primero
-1 = SL alcanzado primero
 0 = ninguna barrera alcanzada
NaN = DIR no operable o ventana futura incompleta
```

Se guarda también:

```text
bar_*_event_step
```

que indica la barra futura 1...H en la que ocurre TP o SL.

## 5.2. Configuraciones evaluadas

```text
TP 1,0 / SL 1,0
TP 1,5 / SL 1,0
TP 2,0 / SL 1,0
TP 2,5 / SL 1,0
TP 1,0 / SL 0,5
TP 1,5 / SL 0,5
TP 2,0 / SL 0,5
```

Universo:

```text
9 DIR × 7 configuraciones = 63 targets BAR
63 columnas event_step
```

Dataset guardado:

```text
mnq_targets_bar.parquet
1.024.062 filas
169 columnas al momento del guardado
```

## 5.3. Implementación de barreras

La entrada y las barreras se evalúan exclusivamente con precios `close`.

Para LONG:

```text
TP = close(t) + tp_mult × threshold
SL = close(t) - sl_mult × threshold
```

Para SHORT:

```text
TP = close(t) - tp_mult × threshold
SL = close(t) + sl_mult × threshold
```

No se utilizan `high` ni `low`. Esto mantiene consistencia con S03, pero representa cierres de un minuto, no ejecución intrabar real.

## 5.4. Resultado agregado por configuración

| TP | SL | TP promedio | SL promedio | No event | TP entre resueltos |
|---:|---:|---:|---:|---:|---:|
| 1,0 | 1,0 | 95,18 % | 4,82 % | 0,00 % | 95,18 % |
| 1,5 | 1,0 | 52,73 % | 5,85 % | 41,42 % | 90,13 % |
| 2,0 | 1,0 | 30,26 % | 6,65 % | 63,10 % | 82,20 % |
| 2,5 | 1,0 | 18,02 % | 7,06 % | 74,92 % | 72,07 % |
| 1,0 | 0,5 | 81,07 % | 18,93 % | 0,00 % | 81,07 % |
| 1,5 | 0,5 | 44,39 % | 22,71 % | 32,89 % | 66,29 % |
| 2,0 | 0,5 | 25,21 % | 24,24 % | 50,55 % | 51,08 % |

La notebook selecciona:

```text
TP 1,5 / SL 1,0
```

como configuración intermedia.

Resultados de las cinco familias conservadas:

| Target BAR | Filas operables | TP | SL | No event | TP/resueltos |
|---|---:|---:|---:|---:|---:|
| `bar_p50_h30_tp15_sl10` | 529.982 | 53,60 % | 5,71 % | 40,70 % | 90,38 % |
| `bar_p50_h60_tp15_sl10` | 505.020 | 52,23 % | 5,54 % | 42,23 % | 90,41 % |
| `bar_p50_h90_tp15_sl10` | 482.603 | 52,38 % | 5,93 % | 41,69 % | 89,83 % |
| `bar_p40_h60_tp15_sl10` | 597.024 | 56,30 % | 7,31 % | 36,39 % | 88,51 % |
| `bar_p60_h60_tp15_sl10` | 413.798 | 48,27 % | 4,14 % | 47,58 % | 92,10 % |

## 5.5. Validaciones BAR

Se verificó con cero errores:

```text
63 targets y 63 event_step
valores permitidos {-1, 0, 1, NaN}
BAR solo definido cuando DIR es operable
coherencia de event_step
ventanas completas sin cruzar días
event_step dentro de 1...H
TP 1,0 sin casos no_event
```

## 5.6. Advertencia metodológica central

La dirección de BAR no se define con información disponible en `t`. Se toma de DIR, y DIR fue construido usando toda la misma ventana futura.

Por lo tanto:

```text
futuro → selecciona dirección mediante DIR
futuro → evalúa TP/SL mediante BAR
```

Esto no es leakage de features, porque se trata de una etiqueta. Sin embargo, produce una selección retrospectiva de la dirección y explica la proporción muy alta de TP.

Consecuencias:

- `BAR` no representa el resultado de una dirección elegida ex ante;
- la tasa TP/resueltos no debe interpretarse como win rate de estrategia;
- BAR por sí solo no indica LONG o SHORT;
- su utilidad final es limitada si no se combina con dirección;
- este punto motivó posteriormente la prioridad de OPC frente a BAR.

## 5.7. BAR preseleccionados

```text
bar_p50_h30_tp15_sl10
bar_p50_h60_tp15_sl10
bar_p50_h90_tp15_sl10
bar_p40_h60_tp15_sl10
bar_p60_h60_tp15_sl10
```

---

# 6. Target OPC

## 6.1. Definición

OPC combina DIR y BAR:

```text
DIR = 0                  → NO_TRADE
DIR = ±1 y BAR = 0       → NO_TRADE
DIR = 1  y BAR = 1       → LONG_TP
DIR = 1  y BAR = -1      → LONG_SL
DIR = -1 y BAR = 1       → SHORT_TP
DIR = -1 y BAR = -1      → SHORT_SL
DIR = NaN                → OPC = NaN
```

`NO_TRADE` agrupa dos fenómenos diferentes:

```text
ausencia de dirección significativa
dirección existente, pero sin TP ni SL
```

Esto genera una clase heterogénea que debe considerarse al modelar.

## 6.2. Codificación realmente guardada en S04

```text
0 = NO_TRADE
1 = LONG_TP
2 = LONG_SL
3 = SHORT_TP
4 = SHORT_SL
```

La columna numérica fue creada como `float`, por lo que se guarda como `0.0...4.0`.

Advertencia crítica: esta codificación no coincide con el mapeo planificado posteriormente en Stage_07, donde `NO_TRADE` aparecía como clase 4. Las notebooks de modelado deben leer el metadata real y no asumir una codificación escrita manualmente.

## 6.3. OPC construidos

```text
opc_p50_h30_tp15_sl10
opc_p50_h60_tp15_sl10
opc_p50_h90_tp15_sl10
opc_p40_h60_tp15_sl10
opc_p60_h60_tp15_sl10
```

Distribución:

| Target OPC | Válidas | NaN | NO_TRADE | Resueltos | Ganadores/resueltos |
|---|---:|---:|---:|---:|---:|
| `opc_p50_h30_tp15_sl10` | 979.602 | 4,34 % | 67,92 % | 32,08 % | 90,38 % |
| `opc_p50_h60_tp15_sl10` | 935.142 | 8,68 % | 68,80 % | 31,20 % | 90,41 % |
| `opc_p50_h90_tp15_sl10` | 890.682 | 13,02 % | 68,40 % | 31,60 % | 89,83 % |
| `opc_p40_h60_tp15_sl10` | 935.142 | 8,68 % | 59,39 % | 40,61 % | 88,51 % |
| `opc_p60_h60_tp15_sl10` | 935.142 | 8,68 % | 76,81 % | 23,19 % | 92,10 % |

En los targets p50 existe una inclinación aproximada de 53–54 % hacia SHORT entre operaciones resueltas.

Las clases perdedoras individuales representan aproximadamente 1–2,3 %, generando desbalance severo.

## 6.4. Estabilidad temporal declarada

Para el conjunto OPC analizado:

```text
Development 2020–2024:
NO_TRADE ≈ 72,47 %
Resueltos ≈ 27,53 %
Loser ratio ≈ 2,25 %
Ganadores/resueltos ≈ 92,04 %

Final test 2025–2026:
NO_TRADE ≈ 51,47 %
Resueltos ≈ 48,53 %
Loser ratio ≈ 6,79 %
Ganadores/resueltos ≈ 86,21 %
```

Conclusión: OPC no es estacionario. 2025–2026 es más activo, pero también más riesgoso.

## 6.5. Validaciones OPC

Cero errores en:

```text
existencia de DIR, BAR, label y clase
valores permitidos
mapeo label ↔ clase
consistencia DIR + BAR → OPC
manejo de NaN
DIR=0 → NO_TRADE
BAR=0 → NO_TRADE
ausencia de DIR operable con BAR faltante
```

---

# 7. Preselección final de S04

S04 conserva cinco familias:

```text
p50_h30: DIR + BAR + OPC
p50_h60: DIR + BAR + OPC
p50_h90: DIR + BAR + OPC
p40_h60: DIR + BAR + OPC
p60_h60: DIR + BAR + OPC
```

Total:

```text
5 DIR
5 BAR
5 OPC
15 targets
```

La decisión escrita en S04 fue:

```text
Candidato operativo principal preliminar:
opc_p50_h30_tp15_sl10
```

Esta decisión era descriptiva, no predictiva.

## Decisión posteriormente vigente

Stage_06 concluyó que:

- BAR mostró mayor predictibilidad estadística;
- BAR no es suficiente operativamente porque no contiene dirección;
- OPC debe tener prioridad;
- el target principal para Stage_07 pasa a ser:

```text
opc_p50_h60_tp15_sl10
```

Por lo tanto, el archivo `mnq_targets_stage_decision.json` de S04 contiene una decisión histórica que ya no representa el estado vigente del proyecto.

---

# 8. Artefactos principales

```text
mnq_targets_dir.parquet
mnq_targets_bar.parquet
mnq_targets_bar_summary.csv
mnq_targets_bar_metadata.csv
mnq_targets_bar_metadata.parquet
mnq_targets_opc.parquet
mnq_targets_opc_metadata.parquet
mnq_targets_opc_metadata.csv
mnq_targets_opc_summary.csv
mnq_targets_opc_selected_families.csv
mnq_targets_selected_final.parquet
mnq_targets_selected_final.csv
mnq_targets_selected_metadata.parquet
mnq_targets_selected_metadata.csv
mnq_targets_stage_decision.json
```

Dataset final preseleccionado:

```text
mnq_targets_selected_final
Shape: 1.024.062 × 26
```

Los CSV y Parquet son artefactos locales pesados. No deben cargarse al repositorio de Claude.

---

# 9. Problemas heredados

S04 utiliza los regímenes producidos por S01:

```text
Regular: 10:30–15:29
Closing: 15:30–15:59
16:00 clasificada como Overnight
```

La convención aprobada posteriormente es:

```text
Regular: 10:30–14:59
Closing: 15:00–16:00
```

Por ello:

- thresholds por régimen;
- targets;
- análisis por régimen;
- resultados de Closing;

heredan la definición anterior.

En S04, Closing presenta hasta 99,33 % de NaN promedio por falta de ventana futura, resultado directamente afectado por su duración original de solo 30 minutos.

---

# 10. Reglas vigentes para etapas posteriores

```text
No usar DIR, BAR, OPC ni event_step como features.
No usar columnas futuras auxiliares.
No cruzar días.
Ajustar modelos y transformaciones solo con train.
No asumir el mapeo de clases: leer metadata.
Interpretar el alto TP como propiedad del etiquetado, no como rentabilidad.
Tratar NO_TRADE como una clase heterogénea.
Mantener opc_p50_h60_tp15_sl10 como target principal actual.
Revisar el impacto de corregir regímenes antes de reconstruir todo el pipeline.
```
