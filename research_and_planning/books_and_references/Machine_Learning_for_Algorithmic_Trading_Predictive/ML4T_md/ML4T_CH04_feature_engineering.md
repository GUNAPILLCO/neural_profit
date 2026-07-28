# Capítulo 4 — Financial Feature Engineering: How to Research Alpha Factors

## 1. Conocimiento explícito del libro

- Los alpha factors son transformaciones de datos de mercado, fundamentales o alternativos diseñadas para extraer señales predictivas relacionadas con los retornos.
- El Information Coefficient mide mediante correlación de Spearman la asociación entre el ranking de un factor y los forward returns. Un IC reducido puede resultar útil cuando la estrategia dispone de suficiente amplitud efectiva.
- En Alphalens, factor turnover mide la proporción de activos que cambia de pertenencia a un cuantil entre periodos consecutivos.
- El filtro de Kalman estima un estado latente mediante un modelo dinámico probabilístico y actualizaciones sucesivas a partir de observaciones ruidosas.
- Las wavelets descomponen una señal en componentes de distintas escalas, permitiendo modificar o eliminar determinados coeficientes antes de reconstruirla.
- El capítulo utiliza NumPy, pandas y TA-Lib para crear factores, y Zipline y Alphalens para realizar backtests y evaluaciones principalmente multiactivo.
- Los forward returns pueden alinearse con el instante de la señal mediante desplazamientos temporales. Esta información corresponde exclusivamente al resultado futuro o target.

## 2. Implicaciones validadas para MNQ

- RSI, ATR, Bandas de Bollinger, OBV y otros indicadores son computables con las barras OHLCV actuales. Su utilidad predictiva para MNQ de un minuto permanece pendiente de validación fuera de muestra.
- Los datos futuros empleados para construir targets no deben ingresar en las features ni contaminar el preprocesamiento o los splits temporales.
- El IC temporal puede estudiarse como adaptación para factores y resultados continuos u ordenables de un único instrumento. Su cálculo debe considerar dependencia serial, horizontes solapados, folds y regímenes.
- Una hipótesis económica, conductual o de microdinámica puede ayudar a priorizar features y reducir falsos descubrimientos, sin ser un requisito absoluto para todas ellas.
- La evaluación univariada es diagnóstica: una feature con señal individual débil aún podría aportar mediante interacciones con otras variables.

## 3. Técnicas no transferibles directamente

- El ranking transversal de cientos de activos y la construcción de carteras por cuantiles no se trasladan directamente a un único futuro MNQ.
- El factor turnover de Alphalens no equivale directamente a la variabilidad temporal de una señal ni a la cantidad de operaciones de una estrategia.
- Los factores fundamentales no están disponibles ni son prioritarios en la fase actual.
- Zipline y Alphalens pueden aportar conceptos y métricas, pero parte de su implementación presupone universos multiactivo.

## 4. Preguntas y experimentos pendientes

- Comparar las ventanas de 30, 60 y 90 minutos con otros horizontes mediante evaluación fuera de muestra.
- Evaluar Kalman de manera causal y medir conjuntamente reducción de ruido, retraso y rendimiento predictivo.
- Implementar wavelets utilizando únicamente una ventana histórica que finalice en cada instante de predicción. Aplicarlas sobre un fold completo no evita necesariamente leakage.
- Construir cuantiles temporales mediante distribuciones rolling o expanding basadas exclusivamente en observaciones pasadas.
- Separar persistencia del score, cambios de estado de la señal, turnover operativo y costes financieros.
- Evaluar el IC temporal en distintos horizontes cuando el resultado sea ordenable, ajustando el análisis por dependencia serial y solapamiento.
- Comparar features completas, selección por correlación, selección supervisada y posibles interacciones no lineales.