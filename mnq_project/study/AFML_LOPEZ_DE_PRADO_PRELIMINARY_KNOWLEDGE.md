# CONOCIMIENTO PRELIMINAR DE LÓPEZ DE PRADO PARA EL PROYECTO MNQ

## 1. Propósito

Este documento reúne el conocimiento preliminar más relevante de:

**Marcos López de Prado — _Advances in Financial Machine Learning_ (2018)**

Su función es proporcionar a Claude un contexto técnico compacto antes de completar el estudio detallado del libro capítulo por capítulo.

No reemplaza la lectura del libro ni convierte sus métodos en decisiones automáticas del proyecto. Su objetivo es identificar:

- principios metodológicos;
- estructuras de datos;
- alternativas de labeling;
- controles contra leakage y dependencia temporal;
- técnicas de validación;
- métodos de interpretación de features;
- criterios de backtesting y riesgo;
- features financieras experimentales;
- mejoras de implementación aplicables al proyecto MNQ.

Toda propuesta incluida aquí debe interpretarse como una de las siguientes categorías:

```text
[LIBRO]       método o afirmación presentada por López de Prado;
[PROYECTO]    restricción, resultado o decisión vigente de MNQ;
[HIPÓTESIS]   propuesta todavía no validada;
[EXPERIMENTO] comparación controlada sugerida;
[NO APLICA]   técnica que no puede implementarse fielmente con los datos actuales.
```

La prioridad general es:

```text
causalidad
→ ausencia de leakage
→ robustez fuera de muestra
→ calibración
→ viabilidad económica
→ complejidad del modelo
```

---

## 2. Contexto mínimo del proyecto

```text
Instrumento: Micro E-mini Nasdaq-100 Futures (MNQ)
Datos disponibles: OHLCV de 1 minuto
Periodo aproximado: 2020–2026
Horario objetivo: 04:30–16:00, America/New_York
Ventanas históricas: 30, 60 y 90 minutos
Regímenes: Overnight, Pre-market, Opening, Regular y Closing
```

Targets disponibles o estudiados:

- `DIR`: dirección futura;
- `BAR`: resultado relacionado con barreras;
- `OPC`: clasificación multiclase con `NO_TRADE`, `LONG_TP`, `LONG_SL`, `SHORT_TP` y `SHORT_SL`.

Modelos previstos:

- Dummy y reglas simples;
- Logistic Regression;
- árboles, Random Forest y boosting;
- MLP;
- CNN1D;
- LSTM;
- GRU;
- TCN.

Los datos actuales no incluyen:

- trades individuales;
- ticks;
- bid y ask;
- spread observado;
- profundidad de mercado;
- cancelaciones y modificaciones de órdenes;
- dirección agresora;
- orden temporal entre `high` y `low` dentro de cada minuto;
- slippage real;
- interés abierto intradía validado.

Por ello, varias técnicas del libro solo pueden aplicarse parcialmente o requieren datos futuros de mayor granularidad.

---

# 3. Aporte central del libro

El libro no se centra en elegir un algoritmo específico. Su contribución principal es explicar por qué los métodos estándar de Machine Learning suelen fallar cuando se aplican directamente a datos financieros.

La cadena de investigación propuesta puede resumirse como:

```text
datos financieros correctamente estructurados
→ eventos y labels económicamente coherentes
→ tratamiento de observaciones no IID
→ transformaciones que preserven información
→ validación con purging y embargo
→ importancia y selección de features
→ tuning interno sin contaminar la evaluación
→ predicciones y probabilidades
→ tamaño de posición
→ backtesting con control de selección múltiple
→ medición de riesgo y probabilidad de fracaso
→ despliegue y seguimiento
```

El principio más importante para MNQ es que el modelo no puede evaluarse de forma aislada. Un resultado aparentemente bueno puede ser falso aunque la arquitectura, el código y el backtest parezcan correctos.

El libro debe utilizarse como una guía de **gobernanza científica del pipeline**, no como una colección de recetas para copiar.

---

# 4. Conceptos prioritarios para MNQ

## 4.1. Financial Machine Learning como disciplina específica

### Conocimiento del libro

López de Prado sostiene que los proyectos financieros requieren métodos específicos porque presentan:

- baja relación señal/ruido;
- observaciones dependientes;
- cambios de régimen;
- series no estacionarias;
- labels solapados;
- múltiples intentos de investigación;
- alto riesgo de sobreajuste de backtest.

También propone abandonar el desarrollo aislado de estrategias y organizar la investigación como una cadena de producción con funciones diferenciadas:

- curación de datos;
- análisis de features;
- diseño de estrategias;
- backtesting;
- despliegue;
- supervisión de producción.

### Aplicación al proyecto

Aunque el proyecto sea desarrollado principalmente por una persona con asistencia de Claude, debe conservarse la separación lógica entre etapas:

```text
S00–S02  datos
S03–S04  thresholds y targets
S05      features
S06      datasets y baselines
S07+     secuencias y modelos
posterior calibración, decisión, backtesting y riesgo
```

Claude no debe modificar decisiones upstream desde una notebook downstream.

Cada experimento debe registrar:

- versión de datos;
- versión de target;
- versión de features;
- fold;
- modelo;
- hiperparámetros;
- semilla;
- métricas;
- artefactos;
- resultado, incluso si es negativo.

---

## 4.2. Estructura de datos financieros

### Barras de tiempo

El libro advierte que las barras de tiempo pueden sobremuestrear periodos de baja actividad y submuestrear periodos de alta actividad. También pueden exhibir autocorrelación, heterocedasticidad y distribuciones poco compatibles con supuestos IID.

### Aplicación al proyecto

Las barras de un minuto deben mantenerse como baseline porque:

- son los datos realmente disponibles;
- preservan el horario y los regímenes intradía;
- permiten definir horizontes de 30, 60 y 90 minutos;
- sostienen el pipeline histórico.

La advertencia del libro no implica reemplazarlas automáticamente.

### Barras de ticks, volumen, dólares, imbalance y runs

```text
[NO APLICA FIELMENTE]
```

Las barras de ticks y las barras impulsadas por información requieren operaciones individuales y, en varios casos, dirección de los trades.

Con OHLCV de un minuto solo pueden construirse aproximaciones de barras de volumen o valor negociado. Estas aproximaciones presentan:

- overshoot del umbral;
- imposibilidad de dividir correctamente la barra de un minuto;
- pérdida del instante exacto del cruce;
- menor fidelidad respecto del método original.

### Experimento posible

Comparar barras de un minuto con aproximaciones de volumen o valor negociado mediante:

- frecuencia de barras por jornada;
- autocorrelación de retornos;
- heterocedasticidad;
- cobertura de regímenes;
- balance de clases;
- cantidad de muestras;
- desempeño fuera de muestra bajo el mismo protocolo.

No seleccionar una estructura solamente porque sus retornos parezcan más normales.

---

## 4.3. Tratamiento de contratos futuros y rollover

### Conocimiento del libro

El cambio entre contratos puede introducir saltos que no representan un movimiento económico real. El libro presenta:

- `Single Future Roll` para un futuro individual;
- `ETF Trick` para series multiproducto, spreads o carteras con pesos variables.

### Aplicación al proyecto

Antes de ajustar precios debe auditarse si alguna:

- feature;
- secuencia;
- ventana de 30, 60 o 90 minutos;
- label;
- operación simulada;

atraviesa un cambio de contrato.

Posibles soluciones, en orden de simplicidad:

1. reiniciar cálculos por contrato;
2. excluir ventanas que crucen el rollover;
3. construir una serie continua ajustada;
4. utilizar `Single Future Roll` si se requiere continuidad;
5. reservar el `ETF Trick` para extensiones multiproducto.

Si se genera una serie ajustada deben conservarse dos representaciones:

```text
precio ajustado o retorno continuo
→ análisis, features y P&L continuo;

precio crudo negociable
→ entrada, salida, tamaño, margen y costes.
```

Los precios ajustados nunca deben tratarse como precios reales de ejecución.

---

## 4.4. Muestreo basado en eventos y filtro CUSUM

### Conocimiento del libro

El filtro CUSUM selecciona observaciones cuando el movimiento acumulado desde el último reinicio supera un umbral. Busca evitar la generación repetida de eventos cuando el precio oscila alrededor de un nivel.

### Aplicación al proyecto

CUSUM puede utilizarse para comparar dos paradigmas:

```text
predicción en cada minuto
vs
predicción únicamente ante eventos relevantes
```

Variables posibles para el filtro:

- cambios de log-precio;
- retornos acumulados;
- volatilidad;
- rango;
- volumen relativo;
- estadísticos de quiebre estructural.

El umbral debe calibrarse únicamente dentro del train y puede depender de:

- volatilidad;
- régimen;
- minuto del día;
- contrato o periodo.

CUSUM no debe sustituir el dataset base sin demostrar mejora fuera de muestra.

---

## 4.5. Labeling con horizonte fijo y thresholds dinámicos

### Conocimiento del libro

El labeling a horizonte fijo puede ser problemático cuando utiliza el mismo umbral en contextos de volatilidad muy diferentes. El libro propone thresholds relacionados con el riesgo observado.

### Aplicación al proyecto

Los targets `DIR` pueden mantenerse como baselines, pero deben compararse variantes con:

- thresholds por régimen;
- thresholds por volatilidad rolling;
- thresholds por minuto del día;
- thresholds relativos a ATR o rango;
- horizontes de 30, 60 y 90 minutos.

Los thresholds deben estimarse con datos permitidos del train. No deben recalibrarse observando el fold externo.

---

## 4.6. Triple-barrier method

### Conocimiento del libro

El método de triple barrera determina el resultado por la primera barrera alcanzada:

- barrera superior de beneficio;
- barrera inferior de pérdida;
- barrera vertical de tiempo máximo.

El label depende de toda la trayectoria entre el inicio y el primer toque.

### Aplicación al proyecto

Los targets `BAR` y `OPC` deben auditarse contra una definición formal de triple barrera:

```text
evento inicial t0
side disponible en t0, si existe
nivel de entrada
profit taking
stop loss
barrera vertical t1
primera barrera tocada
retorno realizado
label final
```

### Restricción crítica de OHLCV de un minuto

Cuando `high` y `low` alcanzan TP y SL dentro de la misma barra no se conoce cuál ocurrió primero.

Se deben comparar políticas explícitas:

- peor caso;
- mejor caso solo como cota superior, no como resultado principal;
- marcar como ambiguo;
- excluir el evento;
- resolver con datos más granulares si se adquieren.

Evaluar barreras solamente con `close` produce un target de cierres, no una simulación de ejecución intraminuto.

### Regla sobre la dirección

Si la dirección no está disponible causalmente en `t0`, no debe construirse retrospectivamente usando el movimiento futuro.

Las alternativas válidas son:

1. aprender la dirección con barreras simétricas;
2. definir la dirección mediante una regla primaria causal;
3. utilizar meta-labeling para decidir operar o no operar una señal primaria.

---

## 4.7. Meta-labeling

### Conocimiento del libro

Meta-labeling separa dos decisiones:

```text
modelo primario
→ propone side: long o short;

modelo secundario
→ decide aceptar, rechazar o dimensionar esa oportunidad.
```

El modelo secundario utiliza labels binarios y busca filtrar falsos positivos del modelo primario.

### Aplicación al proyecto

Una arquitectura posible es:

```text
Modelo primario:
DIR, momentum, reversión, breakout o clasificador direccional

Modelo secundario:
TRADE / NO_TRADE

Salida:
side × probabilidad de aceptación × regla de tamaño
```

`OPC` no debe denominarse meta-labeling solamente por contener clases long, short y no-trade. Para ser meta-labeling debe existir una side primaria causal y externa al modelo secundario.

### Experimentos posibles

- modelo direccional único frente a sistema primario + meta-modelo;
- modelo primario de alto recall y filtro secundario de precisión;
- meta-modelos separados para long y short;
- meta-modelos separados por régimen;
- comparación de F1, cobertura, expectativa y costes.

---

## 4.8. Pesos por solapamiento, concurrencia y uniqueness

### Conocimiento del libro

Los labels financieros suelen compartir parte del mismo camino de precios. Dos muestras cercanas no representan necesariamente dos observaciones independientes.

El libro propone medir:

- número de labels concurrentes;
- uniqueness promedio de cada evento;
- pesos por contribución de retorno;
- time decay;
- sequential bootstrap.

### Aplicación al proyecto

Con eventos cada minuto y horizontes de 30, 60 o 90 minutos, el solapamiento puede ser alto.

Para cada evento se debe almacenar:

```text
t0 = inicio del evento
t1 = final real del label
concurrency = número de eventos activos durante su vida
average_uniqueness = información temporal exclusiva
```

### Experimentos posibles

Comparar:

1. pesos uniformes;
2. class weights;
3. pesos por uniqueness;
4. uniqueness × retorno absoluto;
5. uniqueness × time decay;
6. sequential bootstrap para bagging.

No se deben combinar automáticamente class weights, sample weights y oversampling sin analizar sus efectos conjuntos.

La evaluación debe incluir:

- clases minoritarias;
- calibración;
- estabilidad por fold;
- sensibilidad a observaciones extremas;
- desempeño económico.

---

## 4.9. Diferenciación fraccional

### Conocimiento del libro

Existe una tensión entre:

```text
precios
→ conservan memoria, pero suelen ser no estacionarios;

retornos
→ son más estacionarios, pero eliminan gran parte de la memoria de niveles.
```

La diferenciación fraccional busca encontrar el menor orden `d` que produzca una serie suficientemente estacionaria conservando la mayor memoria posible.

### Aplicación al proyecto

Puede incorporarse como un bloque experimental de features:

- `fracdiff(close)`;
- `fracdiff(log_close)`;
- diferenciación fraccional de VWAP aproximado o medias;
- variantes por régimen o jornada.

### Reglas obligatorias

- estimar `d` únicamente con train;
- utilizar ventana fija para evitar longitud creciente y leakage operativo;
- conservar metadata del orden y del ancho de ventana;
- comparar con retornos y diferencias simples;
- no mezclar contratos sin auditar rollover;
- definir si la memoria se reinicia por jornada o utiliza una serie continua.

La estacionariedad estadística no demuestra capacidad predictiva. El criterio final es la mejora estable fuera de muestra.

---

## 4.10. Ensemble methods

### Conocimiento del libro

El libro analiza bagging, Random Forest y boosting desde la perspectiva de:

- bias;
- variance;
- ruido irreducible;
- redundancia de observaciones;
- dependencia entre árboles.

En finanzas, el bagging puede ser más robusto frente al ruido, mientras que boosting puede concentrarse excesivamente en observaciones difíciles o ruidosas.

### Aplicación al proyecto

Los modelos de árboles deben mantenerse como baselines obligatorios antes de redes profundas.

Experimentos prioritarios:

- Random Forest estándar;
- Random Forest con `max_samples` relacionado con uniqueness;
- Extra Trees;
- HistGradientBoosting, LightGBM o XGBoost;
- bagging con sequential bootstrap;
- class weights y sample weights como variantes separadas.

El resultado in-sample y el OOB score no sustituyen una evaluación temporal purgada.

---

## 4.11. Cross-validation financiera: purging y embargo

### Conocimiento del libro

K-Fold estándar falla cuando:

- los labels se solapan;
- el train contiene información derivada del periodo de validación;
- las observaciones se mezclan aleatoriamente;
- el proceso temporal no es intercambiable.

El libro propone:

- purging: eliminar del train eventos cuyo intervalo de label se superpone con el test;
- embargo: excluir una franja posterior al test para reducir dependencia residual.

### Aplicación al proyecto

El walk-forward anual debe conservarse como evaluación externa, pero se debe auditar el límite entre train y validación.

Ejemplo:

```text
si un evento comienza al final de 2021
y su target termina en 2022,
debe eliminarse del train de WF_01.
```

El purging debe utilizar el intervalo real `[t0, t1]`, no una cantidad fija de filas cuando la duración es variable.

El embargo es especialmente relevante en:

- Purged K-Fold;
- CPCV;
- tuning donde el train puede incluir datos posteriores al bloque de validación;
- datasets con features o labels de memoria extensa.

### Distinción necesaria

No confundir:

- lookback de features;
- horizonte del label;
- embargo de validación.

Cada uno controla un mecanismo diferente.

---

## 4.12. Feature importance

### Conocimiento del libro

El libro distingue:

- MDI: Mean Decrease Impurity;
- MDA: permutation importance;
- SFI: Single Feature Importance;
- features ortogonales;
- análisis de sustitución entre variables correlacionadas.

### Aplicación al proyecto

Las features técnicas y de nivel pueden ser altamente redundantes. Un ranking único no debe interpretarse como verdad causal.

### Protocolo recomendado

1. calcular importancia únicamente dentro del train;
2. utilizar validación purgada para MDA y SFI;
3. analizar estabilidad por fold y régimen;
4. agrupar features correlacionadas;
5. realizar ablaciones por familias;
6. comparar niveles absolutos con variables relativas;
7. verificar si una feature actúa como proxy de año, contrato o nivel nominal.

### Riesgos

- MDI favorece variables con muchos valores y puede repartir o esconder importancia entre sustitutos;
- permutation importance puede subestimar features redundantes;
- SFI ignora interacciones;
- PCA u ortogonalización mejoran la geometría estadística, pero reducen interpretabilidad.

La importancia debe utilizarse para formular hipótesis y reducir redundancia, no para afirmar causalidad.

---

## 4.13. Tuning con validación financiera

### Conocimiento del libro

El libro recomienda integrar el tuning con esquemas de CV adecuados a finanzas y destaca:

- grid search;
- randomized search;
- distribuciones log-uniformes para parámetros de escala;
- elección correcta de scoring.

### Aplicación al proyecto

El tuning debe realizarse exclusivamente dentro del train general de cada fold.

```text
train general
→ división temporal o CV purgada interna
→ tuning y early stopping
→ congelamiento
→ reentrenamiento permitido
→ evaluación externa una sola vez
```

No se debe utilizar el fold externo para:

- seleccionar hiperparámetros;
- elegir features;
- elegir lookback;
- definir class weights;
- ajustar thresholds de probabilidad;
- decidir early stopping.

### Scoring

Para clasificación desbalanceada deben compararse:

- Macro-F1;
- balanced accuracy;
- log loss;
- Brier Score cuando corresponda;
- precision y recall por clase;
- cobertura de operaciones.

No utilizar accuracy como único criterio.

---

## 4.14. Bet sizing a partir de probabilidades

### Conocimiento del libro

El tamaño de posición puede depender de:

- probabilidad predicha;
- número de clases;
- señales activas y solapadas;
- discretización del tamaño;
- distancia al precio objetivo.

### Aplicación al proyecto

El sizing no debe implementarse mientras las probabilidades no estén calibradas.

Pipeline requerido:

```text
predicciones OOS
→ calibración dentro de train
→ evaluación de confiabilidad
→ regla de aceptación
→ tamaño de posición
→ límites de exposición
→ backtest con costes
```

Experimentos posibles:

- tamaño fijo;
- trade/no-trade por threshold;
- tamaño proporcional a confianza calibrada;
- tamaño discretizado;
- promedio de señales activas;
- límites por régimen, volatilidad y drawdown.

No utilizar Kelly completo sin estimaciones robustas de probabilidad y payoff.

---

## 4.15. Peligros del backtesting

### Conocimiento del libro

El libro enfatiza que un backtest aparentemente correcto puede ser falso por:

- selección múltiple;
- data snooping;
- leakage;
- sesgo de supervivencia;
- costes omitidos;
- implementación irrealista;
- selección del mejor resultado entre muchos intentos.

El backtesting no debe utilizarse como generador continuo de ideas hasta encontrar una curva atractiva.

### Aplicación al proyecto

Se debe separar:

```text
investigación de features y modelos
→ métricas predictivas OOS;

diseño de reglas operativas
→ congelamiento;

backtest
→ evaluación económica final del sistema congelado.
```

Debe registrarse el número de variantes intentadas:

- targets;
- ventanas;
- regímenes;
- feature sets;
- modelos;
- seeds;
- hiperparámetros;
- reglas de decisión;
- costes.

El mejor resultado observado no puede interpretarse sin considerar todos los intentos anteriores.

---

## 4.16. Walk-forward y Combinatorial Purged Cross-Validation

### Conocimiento del libro

El walk-forward reproduce una secuencia histórica única, pero presenta limitaciones:

- evalúa un solo camino;
- puede depender excesivamente de una trayectoria particular;
- produce pocas observaciones de desempeño por régimen histórico.

CPCV construye múltiples combinaciones purgadas para obtener una distribución de caminos de backtest.

### Aplicación al proyecto

El walk-forward expansivo actual debe mantenerse como protocolo principal inicial porque es fácil de interpretar y reproduce el despliegue temporal.

CPCV puede incorporarse posteriormente para:

- estimar dispersión de resultados;
- medir dependencia de una trayectoria particular;
- comparar estrategias ya congeladas;
- evaluar probabilidad de sobreajuste.

No debe implementarse antes de resolver:

- `t0` y `t1` de cada label;
- purging correcto;
- embargo;
- target definitivo;
- costes y reglas operativas.

CPCV complementa, pero no reemplaza automáticamente, la prueba final en un periodo futuro completamente reservado.

---

## 4.17. Backtesting sobre datos sintéticos

### Conocimiento del libro

El capítulo estudia reglas óptimas sobre procesos sintéticos, especialmente procesos con reversión a la media, para analizar cómo los parámetros del proceso afectan TP, SL y horizonte.

### Aplicación al proyecto

Su utilidad directa para MNQ es limitada porque MNQ no puede asumirse como un proceso Ornstein-Uhlenbeck estable.

Puede utilizarse para:

- validar el motor de triple barrera;
- probar que el pipeline recupera relaciones conocidas;
- estudiar sensibilidad de TP/SL/horizonte;
- simular cambios de régimen;
- medir errores introducidos por barras OHLC de un minuto;
- comparar políticas ante ambigüedad intrabar.

Un buen resultado sobre datos sintéticos valida la implementación, no demuestra alpha real.

---

## 4.18. Estadísticas de backtest

### Conocimiento del libro

El libro amplía la evaluación más allá del Sharpe ratio e incluye:

- características generales de la estrategia;
- frecuencia y duración de apuestas;
- turnover y exposición;
- concentración de retornos;
- drawdown;
- time under water;
- implementation shortfall;
- Probabilistic Sharpe Ratio;
- Deflated Sharpe Ratio;
- métricas de clasificación;
- atribución temporal y por grupos.

### Aplicación al proyecto

Todo backtest de MNQ debe reportar al menos:

```text
P&L bruto y neto
comisiones
spread y slippage asumidos
turnover
número de trades
exposición
win rate
payoff medio positivo y negativo
expectancy
Sharpe y Sortino
drawdown máximo
time under water
Calmar
concentración temporal de retornos
resultados por fold, año, mes y régimen
```

### PSR y DSR

- PSR estima la probabilidad de que el Sharpe supere un benchmark considerando no normalidad y longitud de muestra.
- DSR corrige además por selección múltiple y número de estrategias probadas.

Estas métricas deben calcularse sobre retornos OOS y con el historial de experimentos disponible.

---

## 4.19. Riesgo y probabilidad de fracaso de la estrategia

### Conocimiento del libro

El riesgo de una estrategia depende de la combinación entre:

- frecuencia de apuestas;
- probabilidad de éxito;
- payoff positivo;
- pérdida media;
- asimetría del payout;
- objetivo de Sharpe;
- longitud del historial.

### Aplicación al proyecto

Antes de operar se debe estimar, siempre con predicciones OOS:

- tasa de acierto mínima requerida;
- expectativa por trade;
- sensibilidad a costes;
- probabilidad de no alcanzar el objetivo;
- frecuencia esperada de pérdidas consecutivas;
- capital necesario para tolerar drawdowns plausibles.

La Macro-F1 no informa por sí sola sobre estos riesgos.

---

## 4.20. Hierarchical Risk Parity

### Conocimiento del libro

HRP utiliza clustering jerárquico, cuasi-diagonalización y bisección recursiva para asignar riesgo sin invertir directamente una matriz de covarianza inestable.

### Aplicación al proyecto

```text
[BAJA PRIORIDAD ACTUAL]
```

No es directamente necesario para una única estrategia sobre MNQ.

Puede ser útil si el proyecto evoluciona hacia un portafolio de:

- modelos distintos;
- horizontes 30/60/90;
- estrategias long y short separadas;
- regímenes intradía;
- instrumentos adicionales;
- variantes de momentum y reversión.

En ese contexto, los retornos OOS de cada estrategia pueden tratarse como activos para asignar riesgo de forma jerárquica.

---

## 4.21. Quiebres estructurales

### Conocimiento del libro

El libro presenta:

- CUSUM sobre residuos o niveles;
- pruebas Chow-type Dickey-Fuller;
- Supremum ADF;
- detección de comportamiento explosivo;
- pruebas de submartingala y supermartingala.

### Aplicación al proyecto

Features candidatas:

- estadístico CUSUM rolling;
- tiempo desde el último quiebre;
- magnitud del cambio de media o pendiente;
- SADF rolling;
- indicador de comportamiento explosivo;
- cambio de relación precio-volatilidad-volumen;
- transición entre estados de tendencia y reversión.

Posibles usos:

- feature predictiva;
- filtro de operación;
- muestreo de eventos;
- variable de régimen;
- alerta de degradación de modelo.

### Riesgos

- alto coste computacional;
- selección posterior del punto de quiebre;
- thresholds ajustados con periodos externos;
- uso de ventanas que incluyen el futuro;
- confusión entre quiebre estadístico y oportunidad operativa.

Toda prueba debe implementarse de forma rolling y causal.

---

## 4.22. Entropía y complejidad de la serie

### Conocimiento del libro

La entropía mide la cantidad de información o imprevisibilidad de una secuencia. El libro presenta:

- entropía de Shannon;
- estimador plug-in;
- estimadores Lempel-Ziv;
- codificación binaria;
- codificación por cuantiles;
- codificación por sigma.

### Aplicación al proyecto

Features candidatas sobre ventanas de 30, 60 y 90 minutos:

- entropía de signos de retornos;
- entropía de retornos cuantizados;
- entropía de expansión/contracción de rango;
- complejidad Lempel-Ziv;
- redundancia;
- cambio de entropía;
- entropía condicionada por régimen;
- interacción entre entropía, momentum y volatilidad.

Hipótesis posibles:

```text
baja entropía
→ mayor persistencia o estructura repetitiva;

alta entropía
→ menor predictibilidad o posible reversión.
```

Estas relaciones no deben asumirse. Deben validarse por target, régimen y fold.

Los límites de cuantiles o sigma deben ajustarse únicamente con train.

---

## 4.23. Features de microestructura

### Conocimiento del libro

El capítulo cubre modelos basados en:

- secuencias de precios;
- estimación de spread;
- high-low volatility;
- Corwin-Schultz;
- Kyle, Amihud y Hasbrouck lambda;
- PIN y VPIN;
- tamaños de órdenes;
- cancelaciones;
- order flow firmado.

### Aplicación parcial con OHLCV de un minuto

Pueden estudiarse con cautela:

- high-low volatility estimator;
- rango relativo y volatilidad basada en high/low;
- Corwin-Schultz como estimador indirecto de spread;
- Roll model como proxy indirecto bajo sus supuestos;
- Amihud illiquidity aproximada mediante retorno absoluto/valor negociado;
- impacto precio-volumen agregado.

Estas variables son proxies y no sustituyen bid/ask ni order flow real.

### Técnicas no implementables fielmente

```text
[NO APLICA CON LOS DATOS ACTUALES]
```

- tick rule sobre trades individuales;
- Kyle lambda con flujo firmado real;
- Hasbrouck lambda con innovaciones de order flow;
- PIN;
- VPIN real;
- distribución de tamaños de órdenes;
- tasas de cancelación;
- límite frente a mercado;
- serial correlation de order flow firmado.

No se debe afirmar que una feature OHLCV representa informed trading sin datos que lo respalden.

---

## 4.24. Multiprocessing, vectorización y escalabilidad

### Conocimiento del libro

El libro propone diseñar funciones desde el inicio para:

- vectorización;
- partición en átomos y moléculas;
- procesamiento asíncrono;
- reducción de resultados;
- uso eficiente de múltiples núcleos.

### Aplicación al proyecto

Prioridades de optimización:

- triple barrera;
- cálculo de concurrencia y uniqueness;
- diferenciación fraccional;
- pruebas de quiebre estructural;
- feature importance purgada;
- búsqueda de hiperparámetros;
- generación de secuencias;
- backtesting de múltiples escenarios.

### Reglas de implementación

- vectorizar antes de paralelizar;
- evitar copias completas del dataset;
- utilizar generadores, chunks o memoria mapeada;
- controlar nested parallelism;
- fijar semillas;
- registrar número de workers;
- validar igualdad entre versión serial y paralela;
- no sacrificar trazabilidad por velocidad.

El código del libro corresponde a APIs de 2018 y contiene sintaxis antigua. Debe reimplementarse y probarse con las librerías actuales; no debe copiarse literalmente.

---

## 4.25. Optimización combinatoria, computación cuántica y HPC

### Aplicación al proyecto

```text
[BAJA PRIORIDAD]
```

Los capítulos 21 y 22 aportan una visión de optimización combinatoria e infraestructura científica a gran escala, pero no son necesarios para la primera versión del pipeline MNQ.

Conceptos transferibles:

- particionar problemas complejos;
- utilizar formatos eficientes;
- reducir I/O;
- medir escalabilidad;
- mantener separación entre investigación y producción.

La computación cuántica no debe formar parte del alcance actual.

---

# 5. Prioridad preliminar por capítulo

| Capítulo | Tema | Prioridad MNQ | Aplicación principal |
|---:|---|---|---|
| 1 | Financial ML como disciplina | Crítica | Gobernanza, trazabilidad y separación de etapas |
| 2 | Estructuras de datos | Crítica | Barras, rollover, CUSUM y auditoría de futuros |
| 3 | Labeling | Crítica | Thresholds dinámicos, triple barrera y meta-labeling |
| 4 | Sample weights | Crítica | Solapamiento, uniqueness y sequential bootstrap |
| 5 | Fractional differentiation | Alta experimental | Features estacionarias con memoria |
| 6 | Ensembles | Alta | Baselines de árboles y control de varianza |
| 7 | Cross-validation financiera | Crítica | Purging, embargo y prevención de leakage |
| 8 | Feature importance | Crítica | Redundancia, ablaciones y selección dentro del train |
| 9 | Hyper-parameter tuning | Alta | Tuning interno con CV financiera |
| 10 | Bet sizing | Alta, etapa posterior | Uso de probabilidades calibradas |
| 11 | Peligros del backtesting | Crítica | Control de selección múltiple y realismo |
| 12 | Backtesting por CV | Alta | Walk-forward, Purged CV y CPCV |
| 13 | Datos sintéticos | Media experimental | Validación del pipeline y sensibilidad TP/SL |
| 14 | Estadísticas de backtest | Crítica | DSR, PSR, drawdown, costes y atribución |
| 15 | Strategy risk | Alta | Probabilidad de fracaso y capital requerido |
| 16 | HRP | Baja actual / media futura | Portafolio de estrategias o instrumentos |
| 17 | Structural breaks | Alta experimental | Features de régimen y muestreo de eventos |
| 18 | Entropy features | Media-alta experimental | Predictibilidad y complejidad rolling |
| 19 | Microstructure | Selectiva | Proxies OHLCV; técnicas granulares no disponibles |
| 20 | Multiprocessing | Alta | Escalabilidad reproducible |
| 21 | Quantum/brute force | Baja | Investigación futura |
| 22 | HPC technologies | Baja-media | Arquitectura futura de gran escala |

---

# 6. Experimentos preliminares sugeridos

## Experimento A — Auditoría de rollover

Determinar si las ventanas de features, secuencias o targets atraviesan cambios de contrato.

Comparar:

```text
sin ajuste
vs
reinicio por contrato
vs
exclusión de ventanas
vs
serie continua ajustada
```

---

## Experimento B — Labeling base frente a triple barrera formal

Comparar:

```text
DIR a horizonte fijo
BAR actual
OPC actual
triple barrera simétrica
triple barrera con side causal
meta-labeling binario
```

Mantener los mismos timestamps, folds y feature sets cuando sea posible.

---

## Experimento C — Ambigüedad intrabar

Para eventos donde TP y SL aparecen en la misma vela:

```text
peor caso
exclusión
label ambiguo
close-only
```

Medir cuánto cambian:

- distribución de clases;
- métricas predictivas;
- P&L;
- estabilidad por régimen.

---

## Experimento D — Uniqueness y sample weights

Comparar pesos uniformes, class weights, uniqueness y combinaciones controladas.

Evaluar también si el beneficio proviene de reducir dependencia o solamente de alterar la distribución de clases.

---

## Experimento E — Purging en walk-forward

Medir cuántas observaciones deben eliminarse en cada límite de fold según su `t1` real.

Verificar por separado:

- labels de 30 minutos;
- labels de 60 minutos;
- labels de 90 minutos;
- barreras con duración variable.

---

## Experimento F — Diferenciación fraccional

Comparar como features:

```text
nivel crudo
retorno
log-retorno
diferencia simple
fracdiff con distintos d
```

Seleccionar `d` dentro del train y evaluar estabilidad por año y régimen.

---

## Experimento G — Feature importance robusta

Aplicar:

- MDI;
- permutation importance purgada;
- SFI;
- ablación por familias;
- clustering de features correlacionadas.

Comparar estabilidad del ranking entre folds.

---

## Experimento H — Muestreo cada minuto frente a CUSUM

Usar el mismo target y modelo para determinar si el muestreo por eventos mejora:

- independencia;
- balance;
- estabilidad;
- Macro-F1;
- log loss;
- expectativa neta.

---

## Experimento I — Entropía y quiebres estructurales

Construir bloques separados:

```text
ENTROPY
STRUCTURAL_BREAKS
```

Evaluar cada bloque solo y en combinación con las features base.

---

## Experimento J — Probabilidades y bet sizing

Después de obtener predicciones OOS:

1. calibrar probabilidades dentro de train;
2. comparar thresholds de aceptación;
3. comparar tamaño fijo y variable;
4. incorporar costes;
5. medir DSR y probabilidad de fracaso.

---

# 7. Implicaciones para el pipeline actual

## Datos

- auditar zona horaria, calendario y rollover antes de modelar;
- mantener OHLCV de un minuto como baseline;
- no reconstruir microestructura inexistente;
- separar precios ajustados de precios negociables.

## Targets

- registrar `t0` y `t1` de cada evento;
- garantizar que la side sea causal;
- resolver la ambigüedad TP/SL intrabar;
- comparar thresholds fijos y dinámicos;
- revisar la heterogeneidad de `NO_TRADE`;
- leer el mapping de clases desde metadata, no desde supuestos escritos.

## Features

- calcularse sin cruzar sesiones o contratos prohibidos;
- ajustar transformaciones únicamente dentro del train;
- comparar niveles absolutos con features relativas;
- incorporar fracdiff, entropía y quiebres como bloques experimentales;
- no seleccionar features observando evaluaciones externas.

## Validación

- walk-forward externo;
- tuning interno cronológico o purgado;
- purging por intervalo real del label;
- embargo cuando corresponda;
- mismo conjunto de timestamps para comparar modelos;
- 2026 reservado hasta congelar el pipeline, si continúa intacto.

## Modelos

- comenzar con Dummy, Logistic Regression y árboles;
- avanzar a redes solo si los baselines muestran señal estable;
- no atribuir una mejora a la arquitectura si cambian simultáneamente datos, target o features;
- evaluar estabilidad, no solamente promedio.

## Backtesting

- utilizar predicciones OOS congeladas;
- ejecutar después del cierre de la barra que produce la señal;
- incluir comisiones, spread y slippage;
- modelar posiciones superpuestas;
- reportar resultados por fold, año, mes y régimen;
- registrar todas las variantes probadas;
- utilizar PSR/DSR y análisis de concentración.

---

# 8. Riesgos metodológicos principales

- usar K-Fold aleatorio o shuffle;
- permitir que labels del train terminen dentro del validation;
- ajustar thresholds, escalado, PCA o selección con datos externos;
- construir la dirección con información futura;
- confundir `close` alcanzando una barrera con toque intraminuto;
- ignorar TP y SL simultáneos en una misma vela;
- contar eventos solapados como observaciones independientes;
- aplicar oversampling sin respetar el tiempo;
- interpretar MDI como causalidad;
- utilizar OOB como sustituto de CV temporal;
- seleccionar el mejor de muchos backtests sin corrección;
- utilizar probabilidades no calibradas para sizing;
- ignorar costes y latencia;
- utilizar precios continuos ajustados para ejecución;
- llamar VPIN, order flow o informed trading a proxies OHLCV;
- aplicar fracdiff con `d` estimado sobre todo el periodo;
- calcular cuantiles de entropía con información futura;
- tratar un quiebre detectado retrospectivamente como señal causal;
- copiar código antiguo del libro sin tests y actualización de APIs;
- forzar la aplicación de todos los capítulos.

---

# 9. Reglas para Claude

```text
1. El libro aporta métodos y advertencias, no decisiones automáticas.
2. Antes de entrenar, auditar datos, rollover, targets y tiempos de disponibilidad.
3. Cada label debe incluir t0 y t1.
4. La side debe existir causalmente en t0 o ser aprendida correctamente.
5. Purging se determina por solapamiento de labels, no por intuición.
6. Embargo y lookback no son el mismo concepto.
7. Toda transformación se ajusta únicamente con train.
8. No utilizar shuffle en series financieras.
9. No seleccionar features con el fold externo.
10. Comparar sample weights mediante ablaciones controladas.
11. No asumir que fracdiff, CUSUM o CPCV mejorarán el proyecto.
12. No usar probabilidades para sizing antes de calibrarlas.
13. No confundir clasificación con rentabilidad.
14. No presentar P&L sin costes y reglas causales.
15. Registrar todos los experimentos, incluidos los fallidos.
16. La importancia de una feature no demuestra causalidad.
17. OHLCV de un minuto no permite reconstruir order flow real.
18. Los precios ajustados no son precios de ejecución.
19. Deep Learning debe superar baselines simples bajo los mismos datos y folds.
20. La estabilidad temporal tiene prioridad sobre el mejor resultado aislado.
21. El código del libro debe reimplementarse con tests; no copiarse literalmente.
22. No limitar el proyecto a OPC, a una sola métrica ni a una sola arquitectura.
```

---

# 10. Uso posterior del estudio capítulo por capítulo

Cuando se complete el estudio de cada capítulo, la nota correspondiente debe contener:

```text
fuente y capítulo
objetivo del capítulo
conocimiento explícito del autor
métodos y fórmulas principales
supuestos
conceptos aplicables a MNQ
conceptos parcialmente aplicables
conceptos no aplicables con los datos actuales
cambios propuestos al pipeline
features, targets o métricas candidatas
riesgos de leakage
experimentos controlados
prioridad
acciones para Claude
```

Las notas detalladas deberán corregir este documento cuando:

- una interpretación preliminar sea incorrecta;
- una técnica resulte no transferible;
- aparezca un riesgo no considerado;
- se adopte una decisión formal;
- un experimento produzca evidencia estable;
- se incorporen datos de ticks, bid/ask u order book.

---

# 11. Conclusión preliminar

La principal contribución de _Advances in Financial Machine Learning_ al proyecto MNQ no es una arquitectura predictiva concreta.

Su aporte es construir un marco donde:

```text
los datos financieros
→ se estructuren y auditen correctamente;

los labels
→ representen eventos causales y económicamente coherentes;

las muestras solapadas
→ no se traten como observaciones IID;

la validación
→ evite leakage mediante tiempo, purging y embargo;

las features
→ se interpreten bajo redundancia y estabilidad;

las probabilidades
→ se calibren antes de decidir y dimensionar;

el backtest
→ sea una evaluación final y no una máquina de búsqueda;

el riesgo
→ se mida considerando costes, selección múltiple y probabilidad de fracaso.
```

Para el proyecto MNQ, los capítulos más importantes son los relacionados con:

```text
estructura de datos y rollover
labeling y triple barrera
meta-labeling
sample weights y uniqueness
purging y embargo
feature importance
tuning temporal
backtesting y DSR
quiebres estructurales
entropía
microestructura parcialmente observable
multiprocessing reproducible
```

El estudio posterior deberá convertir estos conceptos en decisiones versionadas, experimentos controlados y código causal antes de utilizarlos para desarrollar modelos de IA o afirmar que existe una estrategia rentable.
