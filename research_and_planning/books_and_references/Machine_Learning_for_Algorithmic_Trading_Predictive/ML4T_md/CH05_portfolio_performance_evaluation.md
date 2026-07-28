# Capítulo 5 — Portfolio Optimization and Performance Evaluation

## 1. Conocimiento explícito del libro

- La evaluación financiera debe considerar conjuntamente retorno y riesgo.
- El Sharpe Ratio mide el exceso de retorno medio por unidad de volatilidad del exceso de retorno.
- El Information Ratio compara el retorno activo respecto de un benchmark con el tracking error frente a dicho benchmark.
- La Ley Fundamental de la Gestión Activa aproxima:

  IR ≈ IC × sqrt(Breadth)

  donde:
  - IC representa la capacidad predictiva;
  - breadth representa el número de apuestas efectivamente independientes.

- La cantidad bruta de señales u operaciones no equivale automáticamente a breadth.
- La anualización mediante la raíz del tiempo supone condiciones cercanas a IID.
- La autocorrelación puede alterar sustancialmente el Sharpe y su anualización.
- La optimización media-varianza es muy sensible a errores en:
  - retornos esperados;
  - volatilidades;
  - covarianzas.
- El portafolio equiponderado 1/N constituye un baseline multiactivo sencillo que, en determinados estudios fuera de muestra, superó optimizadores más complejos.
- El capítulo presenta alternativas como:
  - mínima varianza;
  - Black-Litterman;
  - risk parity;
  - Hierarchical Risk Parity.
- Kelly busca maximizar el crecimiento logarítmico de capital mediante el tamaño de la exposición.
- El libro también menciona Half-Kelly como reducción habitual del sizing para disminuir volatilidad.
- Pyfolio calcula y visualiza métricas de retorno, riesgo, posiciones y transacciones, y permite separar visualmente periodos in-sample y out-of-sample.

## 2. Métricas aplicables al MNQ

La evaluación debe separar:

1. rendimiento predictivo;
2. rendimiento financiero bruto;
3. rendimiento financiero neto de costes.

### Métricas principales

- retorno acumulado;
- retorno medio diario o por sesión;
- volatilidad;
- Sharpe Ratio;
- Sortino Ratio;
- máximo drawdown;
- Calmar Ratio;
- Omega Ratio;
- Tail Ratio;
- asimetría;
- curtosis;
- porcentaje de sesiones positivas;
- mejores y peores sesiones;
- duración y recuperación de drawdowns.

### Interpretación

- Sharpe penaliza toda la volatilidad.
- Sortino considera principalmente desviaciones negativas respecto de un umbral.
- Calmar relaciona rendimiento anualizado y máximo drawdown.
- Omega compara ganancias y pérdidas ponderadas respecto de un retorno objetivo.
- Tail Ratio compara la magnitud de la cola positiva con la cola negativa.
- Ninguna métrica es suficiente por sí sola.

## 3. Granularidad de evaluación

Los resultados deben analizarse separadamente:

- por operación;
- por sesión;
- por día;
- por fold;
- por año;
- por régimen horario;
- por dirección;
- por clase del target.

Las métricas anualizadas de riesgo-retorno deben calcularse preferentemente a partir de retornos diarios o por sesión de la curva de capital.

Los resultados por barra y por operación son diagnósticos complementarios y no deben anualizarse automáticamente.

## 4. Reglas de anualización

No debe calcularse un Sharpe por minuto y multiplicarlo mecánicamente por la raíz del número de minutos anuales.

Antes de anualizar se debe:

- definir una frecuencia estable;
- comprobar autocorrelación;
- documentar el número de periodos utilizado;
- distinguir días sin operaciones;
- mantener consistente el calendario operativo;
- utilizar resultados estrictamente fuera de muestra.

La tasa libre de riesgo puede tener un efecto pequeño en horizontes intradía, pero su omisión debe documentarse en lugar de asumirse automáticamente.

## 5. Benchmarks para MNQ

El Information Ratio solo tiene sentido frente a un benchmark bien definido.

El Nasdaq-100 o una posición pasiva permanente en MNQ no deben utilizarse automáticamente porque la estrategia:

- opera solo durante determinadas horas;
- puede permanecer sin exposición;
- puede alternar posiciones largas y cortas;
- cierra posiciones intradía;
- tiene costes y turnover diferentes.

Benchmarks candidatos:

- no operar o efectivo, para utilidad absoluta;
- regla de clase mayoritaria;
- señal del retorno anterior;
- estrategia simple de momentum;
- estrategia simple de reversión;
- exposición pasiva limitada al mismo horario;
- baseline lineal;
- estrategia con frecuencia y exposición similares.

El Information Ratio debe utilizarse únicamente cuando el benchmark produzca una serie de retornos temporalmente comparable. En otros casos serán más claros Sharpe, P&L neto, drawdown y métricas por operación.

## 6. Drawdowns y riesgo de cola

Debe medirse:

- máximo drawdown;
- fecha del pico;
- fecha del valle;
- profundidad;
- duración;
- tiempo de recuperación;
- tiempo total bajo el máximo histórico;
- peores días;
- peores sesiones;
- peores operaciones.

Pyfolio incluye un Daily VaR simplificado basado en dos desviaciones estándar por debajo de la media diaria.

Para MNQ, esta medida debe complementarse experimentalmente con:

- VaR histórico;
- Expected Shortfall;
- percentiles empíricos;
- análisis de pérdidas extremas;
- periodos de estrés;
- distribución por regímenes.

Estas extensiones son propuestas metodológicas del proyecto, no afirmaciones centrales del capítulo.

## 7. Costes

Los resultados deben presentarse:

- antes de costes;
- después de costes;
- bajo escenarios de sensibilidad.

Deben distinguirse:

- comisión del broker;
- tasas del exchange;
- tasas regulatorias cuando correspondan;
- spread;
- slippage;
- rollover;
- impacto de mercado cuando pueda estimarse.

Spread y slippage no son costes fijos.

## 8. Kelly como técnica futura

Kelly es contenido explícito del capítulo, pero su aplicación actual al MNQ no está justificada.

Requiere estimaciones fiables de:

- probabilidad de cada resultado;
- payoff positivo;
- payoff negativo;
- costes;
- frecuencia de operaciones;
- dependencia entre operaciones;
- estabilidad temporal;
- distribución de resultados.

Para OPC, la existencia de cinco clases complica la fórmula binaria simple.

Las probabilidades calibradas son necesarias, pero no suficientes.

Half-Kelly también aparece explícitamente en el libro. Para el proyecto, su uso sería una aplicación futura y condicionada.

Antes de utilizar Kelly se deberán imponer límites de:

- contratos máximos;
- margen;
- exposición;
- pérdida diaria;
- drawdown;
- concentración por régimen;
- variación máxima del sizing.

## 9. Contenido multiactivo no directamente aplicable

Actualmente no son directamente aplicables:

- mean-variance optimization;
- Black-Litterman;
- 1/N;
- minimum variance;
- risk parity;
- HRP.

Estas técnicas distribuyen capital entre varios activos.

Podrían adquirir relevancia futura si se combinan:

- diferentes instrumentos;
- estrategias económicamente distintas;
- horizontes realmente diferenciados;
- señales con baja dependencia.

No debe asumirse que modelos, regímenes o ventanas del mismo MNQ constituyen activos independientes.

## 10. Uso de pyfolio

Pyfolio puede:

- recibir retornos, posiciones y transacciones;
- calcular métricas;
- visualizar drawdowns;
- comparar con benchmarks;
- separar gráficamente periodos in-sample y out-of-sample;
- analizar distribuciones y periodos de estrés.

Pyfolio no:

- crea los folds walk-forward;
- genera predicciones out-of-sample;
- previene leakage;
- valida el modelo;
- corrige reglas de ejecución;
- garantiza independencia estadística.

La librería puede estar desactualizada; deben conservarse sus principios y métricas, aunque la implementación final utilice herramientas modernas.

## 11. Decisiones pendientes

- Definir la curva de capital oficial del proyecto.
- Elegir la frecuencia base para las métricas.
- Seleccionar benchmarks apropiados.
- Estimar costes reales.
- Definir métricas obligatorias por fold.
- Determinar cómo agregar resultados walk-forward.
- Establecer límites mínimos de operaciones para considerar válida una métrica.
- Analizar autocorrelación de los retornos de estrategia.
- Definir tratamiento de días sin posición.
- Posponer Kelly hasta disponer de probabilidades y payoffs robustos.

## 12. Riesgos metodológicos

- seleccionar el modelo únicamente por Sharpe;
- comparar Sharpe calculados con frecuencias diferentes;
- anualizar retornos de un minuto mecánicamente;
- usar métricas in-sample;
- optimizar umbrales sobre evaluación externa;
- ignorar costes;
- ocultar folds o años negativos mediante agregación;
- considerar cada operación una apuesta independiente;
- usar un benchmark incompatible;
- aplicar Kelly con probabilidades inestables;
- tratar pyfolio como un sistema de validación;
- confundir rentabilidad acumulada con robustez.