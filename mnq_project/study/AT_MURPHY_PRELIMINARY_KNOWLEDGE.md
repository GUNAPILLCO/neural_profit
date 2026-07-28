# 04 — CONOCIMIENTO PRELIMINAR DE MURPHY PARA EL PROYECTO MNQ

## 1. Propósito

Este documento resume únicamente el conocimiento preliminar más relevante de:

**John J. Murphy — _Análisis técnico de los mercados financieros_**

Su función es aportar contexto a Claude antes del estudio detallado capítulo por capítulo con NotebookLM.

El objetivo no es copiar reglas clásicas de trading ni asumir que los patrones descritos por Murphy tienen capacidad predictiva. El objetivo es identificar conceptos que puedan convertirse en:

- features numéricas;
- variables de contexto;
- filtros de operación;
- definiciones de régimen;
- señales auxiliares;
- reglas de riesgo;
- hipótesis y experimentos para los modelos de IA aplicados a MNQ.

Toda propuesta incluida aquí debe considerarse **hipótesis pendiente de validación**.

---

## 2. Contexto mínimo del proyecto

```text
Instrumento: Micro E-mini Nasdaq-100 Futures (MNQ)
Datos: OHLCV de 1 minuto
Periodo aproximado: 2020–2026
Horario: 04:30–16:00, America/New_York
Ventanas históricas: 30, 60 y 90 minutos
Regímenes: Overnight, Pre-market, Opening, Regular y Closing
```

Targets disponibles o estudiados:

- `DIR`: dirección futura;
- `BAR`: resultado de barreras;
- `OPC`: clasificación multiclase con LONG_TP, LONG_SL, SHORT_TP, SHORT_SL y NO_TRADE.

Modelos previstos:

- baselines lineales y árboles;
- MLP;
- CNN1D;
- LSTM;
- GRU;
- TCN.

Los datos actuales no incluyen:

- ticks;
- bid/ask;
- profundidad de mercado;
- dirección agresora;
- secuencia intraminuto;
- interés abierto intradía validado.

---

# 3. Aporte central del libro

Murphy organiza el análisis técnico alrededor de una idea general:

```text
precio + volumen + tiempo
→ tendencia
→ estructura del mercado
→ confirmación o divergencia
→ decisión operativa
```

Para el proyecto MNQ, estos conceptos no deben interpretarse visualmente ni de forma subjetiva. Deben convertirse en definiciones:

```text
objetivas
causales
reproducibles
parametrizadas
evaluables fuera de muestra
```

Una figura, patrón o indicador no debe incorporarse al modelo solamente porque aparezca en el libro.

---

# 4. Conceptos prioritarios para MNQ

## 4.1. Tendencia

Murphy considera la tendencia como el concepto central del análisis técnico.

Una tendencia puede representarse mediante:

- dirección;
- duración;
- pendiente;
- persistencia;
- estructura de máximos y mínimos;
- fuerza;
- aceleración o desaceleración.

### Features candidatas

- pendiente de regresión del cierre;
- pendiente normalizada por ATR o volatilidad;
- retorno acumulado en 30, 60 y 90 minutos;
- relación movimiento neto/rango total recorrido;
- cantidad de cierres consecutivos en la misma dirección;
- proporción de retornos positivos o negativos;
- máximos y mínimos crecientes;
- distancia a medias móviles;
- separación entre medias rápidas y lentas;
- cambio de pendiente;
- duración desde el último cambio de tendencia.

### Uso posible

Estas variables pueden actuar como:

- features predictivas;
- contexto para CNN1D, LSTM, GRU o TCN;
- filtro de tendencia;
- condición para interpretar osciladores;
- variable de régimen.

---

## 4.2. Soporte y resistencia

Murphy define soporte y resistencia como zonas donde la presión compradora o vendedora puede frenar o revertir el precio.

Para ML no deben utilizarse líneas dibujadas manualmente. Deben formalizarse.

### Features candidatas

- distancia al máximo o mínimo de una ventana;
- distancia normalizada por ATR;
- cantidad de contactos anteriores con una zona;
- tiempo desde el último contacto;
- densidad de cierres cerca de un nivel;
- distancia a apertura, máximo y mínimo de la jornada;
- distancia al máximo y mínimo del régimen actual;
- ruptura reciente de soporte o resistencia;
- retorno posterior a una ruptura;
- fallo de ruptura;
- conversión de resistencia en soporte y viceversa.

### Riesgo metodológico

Los extremos deben calcularse únicamente con datos disponibles hasta el instante `t`.

No se debe definir retrospectivamente un nivel utilizando máximos o mínimos futuros.

---

## 4.3. Líneas de tendencia y canales

Las líneas de tendencia y los canales pueden interpretarse como una representación de dirección, pendiente y dispersión.

### Formalizaciones posibles

- regresión lineal rolling;
- bandas alrededor de la regresión;
- canal basado en máximos y mínimos rolling;
- pendiente superior e inferior;
- ancho del canal;
- posición del precio dentro del canal;
- número de rupturas;
- tiempo dentro o fuera del canal;
- cambio de ancho del canal.

### Aplicación

Estas variables pueden ayudar a distinguir:

- tendencia estable;
- tendencia acelerada;
- consolidación;
- ruptura;
- expansión de volatilidad;
- reversión al centro.

---

## 4.4. Retrocesos

Murphy estudia retrocesos porcentuales y relaciones de Fibonacci.

Para el proyecto, la idea relevante no es asumir que 38,2 %, 50 % o 61,8 % son niveles universales, sino medir cuánto retrocede el precio respecto de un impulso previo.

### Features candidatas

- retroceso actual/impulso anterior;
- profundidad del pullback;
- duración del impulso;
- duración del retroceso;
- relación temporal retroceso/impulso;
- recuperación posterior;
- máximo adverso después del impulso.

Los ratios clásicos pueden incluirse como referencias experimentales, no como reglas aceptadas.

---

## 4.5. Rupturas y continuidad

Murphy describe triángulos, rectángulos, banderas, banderines y cuñas como patrones de continuidad o cambio.

Para ML, el concepto más transferible es:

```text
compresión
→ ruptura
→ confirmación o fallo
```

### Features candidatas

- contracción del rango;
- disminución de volatilidad;
- pendiente convergente de máximos y mínimos;
- ancho del rango;
- duración de la consolidación;
- distancia al límite superior e inferior;
- magnitud de la ruptura;
- volumen relativo durante la ruptura;
- retorno después de la ruptura;
- regreso al rango;
- falso breakout.

Los nombres clásicos del patrón son menos importantes que sus componentes geométricos y temporales.

---

## 4.6. Patrones de cambio

Murphy estudia cabeza y hombros, techos y suelos dobles o triples, platillos y púas.

Su utilidad para MNQ es principalmente experimental.

### Posible traducción cuantitativa

- secuencia de máximos o mínimos;
- simetría temporal;
- diferencia de altura entre picos;
- profundidad del valle intermedio;
- pendiente de una línea de cuello;
- volumen relativo por segmento;
- ruptura y posterior confirmación;
- objetivo teórico comparado con movimiento real.

### Limitación

Estos patrones son fáciles de identificar retrospectivamente y difíciles de definir sin ambigüedad.

Deben tratarse como experimentos de baja o media prioridad hasta contar con una definición completamente reproducible.

---

## 4.7. Volumen

Murphy utiliza el volumen como indicador de confirmación.

En MNQ, el volumen de una barra de un minuto puede representar actividad, pero no order flow real.

### Features prioritarias

- volumen relativo por minuto del día;
- volumen relativo dentro del régimen;
- z-score rolling de volumen;
- relación volumen/rango;
- relación volumen/retorno absoluto;
- volumen durante ruptura;
- volumen durante retroceso;
- divergencia entre pendiente de precio y volumen;
- acumulación de volumen en 30, 60 y 90 minutos;
- cambio de volumen respecto de ventanas anteriores.

### Normalización necesaria

El volumen posee una estacionalidad intradía fuerte. Debe normalizarse al menos por:

- minuto del día;
- régimen;
- ventana histórica;
- eventualmente contrato o periodo.

### No disponible actualmente

El interés abierto tratado por Murphy no debe incorporarse hasta disponer de una fuente fiable y alineada temporalmente.

---

## 4.8. Medias móviles

Las medias móviles son relevantes como medidas de tendencia suavizada, no como señales demostradas.

### Features candidatas

- distancia del precio a SMA o EMA;
- pendiente de la media;
- diferencia entre medias rápidas y lentas;
- tiempo desde el último cruce;
- número de cruces recientes;
- dispersión alrededor de la media;
- retorno desde el cruce;
- alineación de varias medias;
- media adaptable como experimento.

Los periodos deben expresarse en minutos y ajustarse a las ventanas de 30, 60 y 90 minutos.

No se deben trasladar automáticamente parámetros diarios clásicos al intradía.

---

## 4.9. Bandas y volatilidad

Murphy incluye bandas de Bollinger y otras envolventes.

### Features candidatas

- ancho de bandas;
- posición normalizada dentro de las bandas;
- distancia a banda superior e inferior;
- ruptura de banda;
- retorno al interior;
- duración fuera de banda;
- compresión previa;
- expansión posterior;
- relación ancho corto/largo.

Estas variables pueden representar:

- volatilidad;
- compresión;
- expansión;
- sobreextensión;
- posible reversión o continuidad.

La dirección posterior debe determinarse empíricamente.

---

## 4.10. Momentum y osciladores

Murphy analiza momentum, ROC, RSI, estocástico, CCI, Williams %R y MACD.

### Uso adecuado

Los osciladores deben entregarse al modelo como variables continuas y contextuales.

No se debe asumir:

```text
RSI > 70 → vender
RSI < 30 → comprar
```

### Features candidatas

- valor actual;
- pendiente;
- aceleración;
- distancia a umbrales;
- tiempo por encima o debajo de umbrales;
- cruces;
- divergencias cuantificadas;
- valor condicionado por tendencia;
- valor condicionado por régimen;
- diferencia entre ventanas.

### Idea relevante de Murphy

Los osciladores deben interpretarse junto con la tendencia. Esto sugiere interacciones como:

```text
momentum × tendencia
RSI × régimen
MACD × volatilidad
estocástico × distancia a soporte
```

---

## 4.11. Divergencias

Una divergencia puede definirse como desacuerdo entre la dirección del precio y la de un indicador.

### Formalización

- pendiente del precio;
- pendiente del indicador;
- signo de ambas pendientes;
- diferencia normalizada;
- duración del desacuerdo;
- extremos locales previos;
- confirmación posterior.

Debe evitarse elegir manualmente los extremos después de observar el resultado.

---

## 4.12. Velas japonesas

Las velas son especialmente compatibles con OHLCV de un minuto porque pueden descomponerse en relaciones numéricas.

### Features prioritarias

- cuerpo = `close - open`;
- cuerpo absoluto;
- rango = `high - low`;
- sombra superior;
- sombra inferior;
- cuerpo/rango;
- cierre dentro del rango;
- apertura dentro del rango anterior;
- gap;
- expansión o contracción;
- dirección de las últimas `n` velas;
- proporción de velas alcistas;
- secuencias de cuerpos y sombras.

Los nombres tradicionales de los patrones pueden conservarse como referencia, pero el modelo debe recibir sus componentes numéricos.

### Limitación

Una vela de un minuto no revela el orden exacto entre high y low.

---

## 4.13. Tiempo, ciclos y estacionalidad

El capítulo de ciclos puede aportar conocimiento útil si se interpreta de forma moderna.

### Aplicaciones posibles

- minuto del día;
- régimen;
- tiempo desde la apertura;
- tiempo hasta el cierre;
- día de la semana;
- semana de vencimiento;
- estacionalidad de volumen;
- estacionalidad de volatilidad;
- periodicidad intradía;
- componentes seno/coseno;
- análisis espectral como experimento.

No se debe asumir que un ciclo histórico permanece estable. Debe evaluarse por año, régimen y contrato.

---

## 4.14. Perspectiva multitemporal

Murphy recomienda analizar el mercado desde horizontes amplios hacia horizontes cortos.

En el proyecto puede traducirse en features simultáneas de:

- 5 minutos;
- 15 minutos;
- 30 minutos;
- 60 minutos;
- 90 minutos;
- jornada acumulada.

El objetivo no es añadir ventanas indiscriminadamente, sino representar contexto y dinámica local.

---

## 4.15. Sistemas de contratación

Murphy distingue entre interpretación discrecional y sistemas mecánicos.

Para Claude, el principio importante es que toda idea debe convertirse en una regla verificable:

```text
datos de entrada
→ cálculo
→ señal
→ entrada
→ salida
→ riesgo
→ costes
→ evaluación
```

Las reglas no deben modificarse después de observar el resultado externo.

---

## 4.16. Gestión monetaria y tácticas

Murphy separa tres componentes:

- pronóstico;
- táctica;
- gestión monetaria.

Para el proyecto significa que la predicción del modelo no es la estrategia completa.

Deben tratarse por separado:

- probabilidad predictiva;
- decisión de operar;
- dirección;
- tamaño;
- stop;
- take profit;
- tiempo máximo;
- costes;
- exposición;
- drawdown.

Una mejora en Macro-F1 no demuestra una mejora económica.

---

## 4.17. Contratos continuos

Murphy discute gráficos continuos de futuros.

Este tema es importante porque los cambios de contrato pueden generar discontinuidades artificiales.

### Aplicación al proyecto

Se debe auditar:

- si las ventanas cruzan contratos;
- si aparecen saltos de precio;
- si los indicadores se contaminan;
- si los targets atraviesan un rollover;
- si las secuencias mezclan contratos.

Si se construye una serie ajustada:

- debe conservarse también la serie cruda;
- la serie ajustada debe usarse para análisis;
- la ejecución debe utilizar precios reales negociables.

---

# 5. Prioridad preliminar por capítulo

## Prioridad alta

- Capítulo 1 — Filosofía del análisis técnico.
- Capítulo 3 — Construcción de gráficos.
- Capítulo 4 — Conceptos básicos de tendencia.
- Capítulos 5 y 6 — Patrones de cambio y continuidad.
- Capítulo 7 — Volumen e interés abierto.
- Capítulos 9 y 10 — Medias móviles, momentum y osciladores.
- Capítulo 12 — Velas japonesas.
- Capítulo 15 — Ordenadores y sistemas de contratación.
- Capítulo 16 — Gestión monetaria y tácticas.
- Apéndice C — Creación de un sistema.
- Apéndice D — Contratos de futuros continuos.

## Prioridad media

- Capítulo 2 — Teoría de Dow.
- Capítulo 8 — Gráficos a largo plazo.
- Capítulo 14 — Ciclos temporales.
- Capítulo 17 — Análisis entre mercados.
- Apéndice B — Market Profile.

## Prioridad baja o experimental

- Capítulo 11 — Puntos y figuras.
- Capítulo 13 — Ondas de Elliott.
- Capítulo 18 — Indicadores bursátiles generales.
- Apéndice A — Indicadores técnicos avanzados.

La prioridad baja no significa que deban ignorarse. Significa que su transferencia al proyecto MNQ es menos directa o requiere datos adicionales.

---

# 6. Experimentos preliminares sugeridos

## Experimento A — Bloques de features Murphy

Construir familias independientes:

```text
TREND
SUPPORT_RESISTANCE
CHANNELS
BREAKOUT
CANDLE_GEOMETRY
VOLUME_CONTEXT
MOVING_AVERAGES
VOLATILITY_BANDS
MOMENTUM_OSCILLATORS
TIME_CONTEXT
```

Comparar:

1. cada bloque por separado;
2. combinación incremental;
3. conjunto compacto;
4. conjunto completo.

---

## Experimento B — Indicadores continuos frente a reglas clásicas

Comparar:

```text
indicador continuo
vs
regla umbral clásica
vs
indicador condicionado por tendencia y régimen
```

Ejemplos:

- RSI continuo frente a RSI 70/30;
- MACD continuo frente a cruces;
- banda continua frente a ruptura binaria.

---

## Experimento C — Rupturas

Evaluar si la combinación:

```text
compresión + ruptura + volumen relativo
```

aporta más información que cualquiera de sus componentes por separado.

---

## Experimento D — Features relativas frente a niveles absolutos

Priorizar variables como:

- distancia normalizada;
- pendiente;
- ratio;
- posición;
- cambio.

Compararlas con niveles nominales para detectar posibles proxies de año o contrato.

---

## Experimento E — Estabilidad

Toda feature derivada de Murphy debe evaluarse por:

- fold;
- año;
- régimen;
- contrato;
- clase;
- ventana de 30, 60 y 90 minutos.

---

# 7. Riesgos metodológicos

- identificación retrospectiva de patrones;
- elección subjetiva de máximos y mínimos;
- optimización excesiva de ventanas;
- uso de parámetros clásicos sin validación;
- múltiples comparaciones;
- selección de features utilizando años externos;
- leakage por extremos futuros;
- normalización global;
- confusión entre volumen y order flow;
- cruce de días o contratos;
- interpretación económica posterior al resultado;
- supervivencia aparente de patrones por data snooping.

---

# 8. Reglas para Claude

```text
1. Murphy aporta hipótesis de mercado, no evidencia predictiva.
2. Toda idea visual debe traducirse a una definición computable.
3. Toda feature debe indicar cuándo queda disponible.
4. Los parámetros deben ajustarse únicamente dentro del train.
5. Los patrones deben compararse con baselines simples.
6. Deben realizarse ablaciones por familias.
7. La estabilidad temporal importa más que un único resultado alto.
8. El volumen de 1 minuto no representa order flow real.
9. El interés abierto no debe usarse sin una fuente validada.
10. Los precios continuos ajustados no son precios de ejecución.
11. No limitar el proyecto a OPC ni a un único modelo.
12. No adoptar todas las técnicas del libro por obligación.
```

---

# 9. Conclusión preliminar

La mayor contribución de Murphy al proyecto MNQ es proporcionar un vocabulario estructurado para describir:

```text
tendencia
estructura local
niveles
rupturas
momentum
volatilidad
volumen
tiempo
táctica
riesgo
```

El trabajo posterior consiste en convertir ese conocimiento en variables causales y experimentos reproducibles.

El estudio capítulo por capítulo con NotebookLM deberá:

- corregir o ampliar este documento;
- extraer fórmulas y reglas concretas;
- identificar conceptos no transferibles;
- proponer features y experimentos;
- registrar riesgos;
- preparar conocimiento compacto para Claude.
