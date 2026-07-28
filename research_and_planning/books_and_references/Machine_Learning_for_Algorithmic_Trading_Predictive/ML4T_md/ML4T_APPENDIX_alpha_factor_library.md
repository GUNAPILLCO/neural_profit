# Apéndice — Alpha Factor Library

## 1. Conocimiento explícito del libro

- El apéndice presenta un catálogo amplio de indicadores implementados mediante TA-Lib, incluyendo medias móviles, momentum y fuerza de tendencia, volumen, volatilidad, ciclos y funciones estadísticas.
- Los 101 Formulaic Alphas de WorldQuant fueron diseñados principalmente para estrategias multiactivo con periodos medios de mantenimiento aproximados de 0,6 a 6,4 días.
- Muchas fórmulas combinan funciones temporales con rankings transversales, VWAP, volumen promedio en dólares, múltiples activos o agrupaciones sectoriales.
- Cuando el VWAP no está disponible, el libro utiliza `(O + H + L + C) / 4` como aproximación muy rudimentaria para demostrar las fórmulas. Esta expresión no es un VWAP real ni está ponderada por volumen.
- El apéndice compara IC, Mutual Information, importancia de variables y SHAP como métodos complementarios de evaluación.
- La correlación de 0,16 entre los rankings de MI e IC corresponde únicamente al experimento realizado en el apéndice y no constituye una propiedad general de estas métricas.

## 2. Factores computables con los datos actuales

Los siguientes indicadores pueden calcularse directamente a partir de las barras OHLCV de un minuto:

- RSI, MACD y Williams %R.
- Bandas de Bollinger y medias móviles.
- ATR y otras medidas derivadas del rango.
- Parabolic SAR y Ultimate Oscillator.
- ADX, PLUS_DI y MINUS_DI.
- OBV y Chaikin A/D.
- Algunas funciones estadísticas móviles y transformaciones de precio.

La posibilidad de calcular un indicador no implica que posea capacidad predictiva para MNQ. Su utilidad deberá validarse fuera de muestra.

Una fórmula de WorldQuant solo puede reproducirse directamente cuando su expresión completa utiliza datos disponibles y no depende de:

- rankings transversales;
- VWAP real;
- `adv(d)`;
- sectores o industrias;
- múltiples activos.

Modificar estos componentes produce una nueva feature experimental, no el alpha original.

## 3. Factores no transferibles directamente

- Los rankings transversales requieren comparar múltiples activos en el mismo instante y no se aplican directamente a un único futuro.
- Los factores Fama-French existen como series externas, pero no forman parte del dataset actual ni pueden derivarse del OHLCV del MNQ.
- Las normalizaciones sectoriales no son aplicables a un único futuro de índice.
- El uso convencional de Alphalens, basado en carteras y cuantiles transversales, no se transfiere directamente al MNQ.

## 4. Métodos de evaluación

- El IC del apéndice evalúa principalmente la asociación transversal entre el ranking de factores y los retornos futuros.
- Para MNQ puede experimentarse con un IC temporal cuando la feature y el resultado sean continuos u ordenables. Esta adaptación debe controlar dependencia serial y horizontes solapados.
- Mutual Information puede detectar asociaciones no lineales, pero su estimación depende de la muestra y de los parámetros utilizados.
- La importancia de variables de LightGBM y los valores SHAP permiten diagnosticar el uso de las features dentro de modelos multivariados.
- Ninguna métrica debe utilizarse como criterio único de selección.
- La estabilidad debe revisarse entre folds, años y regímenes horarios.

## 5. Experimentos pendientes

- Crear rankings y cuantiles temporales mediante ventanas rolling o expanding que finalicen en el instante de predicción.
- Comparar el valor actual de indicadores con su distribución histórica reciente.
- Evaluar si `(O + H + L + C) / 4` aporta información adicional frente al cierre, sin interpretarlo como VWAP real.
- Comparar:
  - dataset completo;
  - eliminación por correlación;
  - selección mediante MI;
  - importancia multivariada;
  - estabilidad de SHAP;
  - rendimiento fuera de muestra.
- Analizar operadores temporales de WorldQuant como `ts_max`, `ts_stddev`, `ts_argmax` y `ts_correlation`, tratándolos como inspiración para nuevas features y no como alphas reproducidos directamente.

## 6. Riesgos específicos para MNQ

- OBV y Chaikin A/D son acumulativos. Para MNQ deben evaluarse versiones por sesión, rolling o normalizadas debido a cambios de contrato, cortes de sesión y estacionalidad intradía del volumen. Esta es una inferencia específica del proyecto.
- Probar masivamente indicadores y fórmulas incrementa el riesgo de falsos descubrimientos y sobreajuste por múltiples comparaciones.
- Los rankings temporales no deben utilizar datos posteriores al instante evaluado.
- La validación de una feature debe considerar:
  - estabilidad fuera de muestra;
  - magnitud y persistencia del efecto;
  - dependencia temporal;
  - múltiples pruebas;
  - interacción con otras variables;
  - costes y utilidad financiera.