# Capítulo 2 — Financial Data Structures

## Objetivo del capítulo

El capítulo explica cómo transformar datos financieros crudos o irregulares en estructuras utilizables por algoritmos de Machine Learning. Sus aportes principales son:

- clasificación de los tipos de datos financieros;
- construcción de barras;
- tratamiento de series formadas por contratos futuros;
- muestreo de observaciones mediante eventos.

Para el proyecto MNQ, los temas prioritarios son la **auditoría de continuidad entre contratos**, la conservación de las barras de un minuto y la evaluación experimental de métodos alternativos de muestreo.

## Conocimiento explícito del autor

### Tipos de datos financieros

El autor distingue cuatro categorías:

1. **Datos fundamentales:** información contable y macroeconómica, expuesta a retrasos de publicación, revisiones y backfilling.
2. **Datos de mercado:** precios, volumen, operaciones, cotizaciones, cancelaciones y demás actividad registrada en mercados o exchanges.
3. **Analíticas:** información previamente procesada por terceros, como recomendaciones, expectativas o señales derivadas.
4. **Datos alternativos:** información primaria que todavía no ha sido incorporada a las fuentes tradicionales, como sensores, imágenes o actividad digital.

El proyecto MNQ utiliza actualmente **datos de mercado agregados en OHLCV de un minuto**.

## Construcción de barras

### Barras de tiempo

Muestrean el mercado a intervalos fijos, por ejemplo cada minuto. El autor señala que pueden:

- sobremuestrear periodos de baja actividad;
- submuestrear periodos de alta actividad;
- presentar autocorrelación;
- presentar heterocedasticidad;
- producir retornos alejados de los supuestos IID utilizados por numerosos métodos estadísticos.

En el proyecto MNQ se mantienen porque:

- son los datos realmente disponibles;
- preservan la hora exacta;
- permiten representar los cinco segmentos intradía;
- son necesarias para horizontes definidos en minutos;
- constituyen la base de los targets y features actuales.

### Barras de ticks

Se generan después de un número determinado de transacciones. Requieren operaciones individuales y no pueden reconstruirse correctamente a partir de OHLCV de un minuto.

### Barras de volumen

Se generan cuando se negocia una cantidad determinada de contratos. El autor las relaciona con mejores propiedades estadísticas que las barras de tiempo.

Con datos de un minuto solo puede construirse una **aproximación**:

- se acumula el volumen de barras consecutivas;
- el límite solo puede detectarse al cierre del minuto;
- no puede dividirse correctamente la barra que supera el umbral;
- aparece un exceso de volumen u overshoot;
- se pierde el instante exacto del cruce.

Por ello, no deben considerarse equivalentes a las barras de volumen construidas desde trades.

### Barras de dólares o valor negociado

Se generan al alcanzar un valor monetario intercambiado. Para MNQ podrían aproximarse mediante precio por volumen, pero presentan las mismas limitaciones de resolución y overshoot que las barras de volumen aproximadas.

El libro no demuestra que estas aproximaciones mejoren el proyecto. Deben compararse empíricamente con las barras de un minuto.

### Barras impulsadas por información

Las barras de imbalance y runs requieren secuencias de operaciones individuales, dirección agresora y volumen por trade. No pueden reconstruirse válidamente desde OHLCV de un minuto.

## Tratamiento de contratos futuros

### Riesgo de discontinuidad

Una serie consolidada puede contener saltos cuando se sustituye un contrato por el siguiente, por ejemplo:

```text
MNQU23 → MNQZ23
```

El diferencial entre ambos contratos no representa necesariamente un movimiento económico ocurrido en el mercado. Puede contaminar:

- retornos;
- volatilidad;
- indicadores;
- secuencias;
- labels;
- targets basados en barreras;
- análisis de PnL.

Sin embargo, la contaminación no debe asumirse. Primero debe verificarse:

- la fecha y hora de cada cambio de contrato;
- si el cambio ocurre entre jornadas;
- si las secuencias cruzan jornadas;
- si alguna ventana histórica atraviesa el cambio;
- si los targets de 30, 60 o 90 minutos atraviesan el cambio;
- si las features de nivel o retorno utilizan observaciones de contratos distintos.

### Single Future Roll

Para una serie de un único futuro, el capítulo propone calcular los gaps entre el cierre del contrato saliente y la apertura del contrato entrante, acumularlos y restarlos de la serie de precios.

Este método es más directo para MNQ que el ETF trick, siempre que se determine que el dataset necesita ajuste.

### ETF Trick

El ETF trick permite representar una cesta, un spread o incluso un futuro como un producto cash sin vencimiento. Puede aplicarse a una sola pata, pero no es la primera opción para el proyecto actual porque el método de single future roll es más simple.

Debe reconsiderarse si el proyecto evoluciona hacia:

- spreads entre contratos;
- múltiples futuros;
- carteras;
- pesos variables;
- reinversión explícita y costes de rebalanceo.

### Precios crudos y precios ajustados

Si se requiere ajuste, deben conservarse ambas representaciones:

- **precios ajustados o retornos continuos:** análisis, features, targets y simulación de PnL cuando una ventana atraviesa el roll;
- **precios crudos:** niveles negociables, ejecución, tamaño de posición y consumo de capital.

No deben sustituirse los precios reales del mercado por precios ajustados en la lógica de ejecución.

## Muestreo basado en eventos

### Filtro CUSUM

El filtro CUSUM selecciona una observación cuando el movimiento acumulado desde el último reinicio supera un umbral.

En MNQ puede aplicarse sobre la serie disponible, por ejemplo:

- precio de cierre;
- log-precio;
- retornos acumulados.

Su uso debe especificar claramente:

- variable filtrada;
- unidad del umbral;
- método de calibración;
- si el umbral es fijo o dependiente de volatilidad;
- tratamiento separado o conjunto de los regímenes intradía.

CUSUM:

- no reconstruye información tick;
- no equivale a una barra de imbalance;
- no garantiza reducir ruido;
- modifica la frecuencia y distribución de las muestras;
- puede alterar el balance de clases y la representación de los regímenes.

Debe tratarse como un experimento alternativo, no como sustitución automática del dataset por minuto.

## Aplicación al proyecto MNQ

| Concepto | Aplicación al proyecto | Decisión | Prioridad |
|---|---|---|---|
| Barras de un minuto | Base temporal de features, targets y regímenes intradía. | **MANTENER** | ALTA |
| Auditoría de roll | Verificar si secuencias, features o targets atraviesan cambios de contrato. | **AUDITAR** | ALTA |
| Single Future Roll | Aplicar solamente si la auditoría demuestra contaminación. | **AUDITAR** | ALTA |
| Precios crudos y ajustados | Conservar ambas series si se implementa ajuste. | **INCORPORAR CONDICIONALMENTE** | ALTA |
| CUSUM | Evaluar como muestreo alternativo de eventos. | **EXPERIMENTAR** | MEDIA |
| Barras de volumen aproximadas | Comparar con la base de un minuto reconociendo sus limitaciones. | **EXPERIMENTAR** | BAJA |
| Barras de dólares aproximadas | Comparar solo si existe una hipótesis y protocolo claros. | **EXPERIMENTAR** | BAJA |
| Barras tick, imbalance y runs | Requieren trades individuales y dirección agresora. | **APLICAR MÁS ADELANTE** | BAJA |
| ETF trick | No prioritario para una única serie outright; útil en futuras extensiones. | **APLICAR MÁS ADELANTE** | BAJA |

## Auditoría requerida sobre el dataset actual

La columna `contract` permite identificar los cambios de contrato. La auditoría debe incluir:

1. ordenar cada transición cronológicamente;
2. localizar la última barra del contrato saliente;
3. localizar la primera barra del contrato entrante;
4. medir el gap entre ambos;
5. identificar si el cambio ocurre dentro de una jornada o entre jornadas;
6. comprobar si las secuencias de 30, 60 y 90 minutos cruzan la transición;
7. comprobar si los targets DIR, BAR y OPC cruzan la transición;
8. revisar features calculadas con `shift`, `rolling`, `diff`, retornos o niveles;
9. cuantificar cuántas muestras serían afectadas;
10. decidir entre:
   - excluir ventanas contaminadas;
   - reiniciar cálculos en cada contrato;
   - ajustar la serie;
   - mantener el dataset sin cambios si no existe contaminación.

La solución debe elegirse después de medir el problema, no antes.

## Evaluación experimental de barras alternativas

Las barras aproximadas de volumen o dólares deben compararse con las barras de un minuto mediante:

- número de barras por jornada;
- estabilidad de la frecuencia de muestreo;
- autocorrelación de retornos;
- heterocedasticidad;
- distribución de retornos;
- cobertura de los cinco regímenes horarios;
- balance de clases;
- cantidad de muestras;
- desempeño out-of-sample bajo el mismo protocolo.

La normalidad no debe utilizarse como criterio único para decidir qué estructura es mejor.

## Riesgos metodológicos

- **Ajustar sin necesidad:** modificar toda la serie aunque ninguna muestra atraviese un roll.
- **No ajustar cuando corresponde:** introducir gaps artificiales en retornos, labels o PnL.
- **Usar precios ajustados para ejecución:** trabajar con niveles que no fueron negociables.
- **Confundir aproximaciones con datos tick:** atribuir a barras agregadas propiedades que no pueden conservar.
- **Perder contexto intradía:** utilizar muestreo por actividad sin controlar la representación de franjas horarias.
- **Cambiar simultáneamente muestreo, targets y modelos:** impedir atribuir el efecto de cada modificación.
- **Elegir CUSUM por resultados favorables:** calibrar el umbral repetidamente sobre validación hasta obtener una mejora aparente.

## Tareas pendientes

1. Auditar todas las transiciones de la columna `contract`.
2. Determinar cuántas features, secuencias y etiquetas atraviesan cada roll.
3. Definir si conviene excluir ventanas, reiniciar cálculos o ajustar precios.
4. Mantener los precios crudos aunque se cree una serie ajustada.
5. Diseñar un experimento CUSUM sin modificar todavía el pipeline principal.
6. Definir una calibración del umbral exclusivamente dentro del conjunto de entrenamiento.
7. Considerar barras aproximadas de volumen o dólares solo como experimentos secundarios.
8. Registrar cualquier nueva estructura de barras como una variante independiente del dataset.

## Conceptos no aplicables con los datos actuales

No pueden implementarse fielmente:

- tick bars;
- tick imbalance bars;
- volume imbalance bars;
- dollar imbalance bars;
- tick runs bars;
- volume runs bars;
- dollar runs bars.

Todos requieren información más granular que OHLCV de un minuto.

## Conceptos que se desarrollarán posteriormente

- **Labeling y triple barrera:** Capítulo 3.
- **Pesos por solapamiento:** Capítulo 4.
- **Diferenciación fraccional:** Capítulo 5.
- **Purging y embargo:** Capítulo 7.
- **Quiebres estructurales:** Capítulo 17.
- **Entropía:** Capítulo 18.
- **Features de microestructura derivables de OHLCV o datos más granulares:** Capítulo 19.

## Conclusión

El Capítulo 2 no obliga a reemplazar las barras de un minuto ni a ajustar inmediatamente toda la serie MNQ. Su contribución principal es establecer una auditoría rigurosa de la estructura de datos.

La prioridad es determinar si los cambios de contrato contaminan realmente las features, secuencias o targets. Solo después debe decidirse entre excluir ventanas, reiniciar cálculos por contrato o crear una serie ajustada mediante single future roll.

CUSUM y las barras de actividad aproximadas permanecen como experimentos controlados que deben evaluarse sin alterar el pipeline base.
