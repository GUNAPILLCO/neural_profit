# Capítulo 1 — Machine Learning for Trading: From Idea to Execution

## 1. Conocimiento del libro

- El trading algorítmico automatiza reglas de inversión, mientras que Machine Learning permite aprender patrones a partir de datos para alcanzar objetivos predictivos.
- La gestión activa busca generar alfa: rendimiento superior al benchmark utilizado para evaluar la estrategia.
- La Ley Fundamental de la Gestión Activa relaciona la calidad de los pronósticos y su aplicación:

  IR ≈ IC × √breadth

- El Information Coefficient mide la relación entre los pronósticos y los resultados, normalmente mediante correlación de rango.
- Breadth representa el número de apuestas efectivamente independientes; una mayor frecuencia operativa no implica automáticamente mayor amplitud.
- El flujo ML4T integra datos, ingeniería de factores, modelado, generación de señales, evaluación, gestión de riesgo y ejecución.
- Los datos deben mantenerse point-in-time: cada observación debe contener únicamente información disponible en el momento de la predicción.

## 2. Implicaciones validadas para MNQ

- Utilizar el flujo ML4T como marco de referencia para conectar datos, features, targets, modelos, señales, backtesting y ejecución.
- Emplear IC únicamente en targets continuos o señales ordenables, complementándolo con métricas específicas de clasificación, calibración y rendimiento financiero.
- Verificar que todas las features de cada barra de un minuto se calculen exclusivamente con información disponible hasta ese instante.
- Medir la degradación señal–ejecución causada por comisiones, slippage, reglas de entrada y limitaciones de las barras OHLCV.

## 3. Preguntas y experimentos pendientes

- Definir benchmarks predictivos, operativos y financieros apropiados para una estrategia intradía sobre MNQ.
- Estimar la amplitud efectiva analizando la dependencia entre targets, scores, posiciones y operaciones consecutivas.
- Comparar features respaldadas por una hipótesis económica o conductual con features obtenidas mediante búsqueda empírica, controlando falsos descubrimientos.
- Evaluar la estabilidad de las métricas predictivas por régimen horario. Para señales continuas u ordenables, incluir el IC por régimen.