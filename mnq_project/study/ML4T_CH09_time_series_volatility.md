# Capítulo 9 — Time-Series Models for Volatility Forecasts and Statistical Arbitrage

## 1. Conocimiento explícito del libro

- Las series temporales deben analizarse respetando su orden secuencial.
- Los modelos dinámicos utilizan observaciones pasadas de la propia variable y, opcionalmente, variables contemporáneas o rezagadas para predecir valores futuros.
- El capítulo cubre:
  - diagnóstico temporal;
  - estacionariedad;
  - modelos AR, MA y ARIMA;
  - SARIMAX;
  - ARCH y GARCH;
  - VAR;
  - cointegración;
  - pairs trading.
- Una serie es débilmente estacionaria cuando:
  - su media es constante;
  - su varianza es finita y constante;
  - su autocovarianza depende únicamente del rezago y no del momento concreto.
- El ADF tiene como hipótesis nula la existencia de una raíz unitaria.
- Rechazar la hipótesis nula aporta evidencia contra la raíz unitaria, pero no garantiza ausencia de:
  - cambios estructurales;
  - estacionalidad;
  - heterocedasticidad;
  - cambios de régimen.
- Los logaritmos pueden estabilizar la escala y las diferencias logarítmicas producen retornos, pero no garantizan estacionariedad.
- ACF y PACF ayudan a diagnosticar dependencias lineales y proponer órdenes AR y MA.
- Ljung-Box comprueba si queda autocorrelación residual hasta determinados rezagos.
- Un resultado no significativo de Ljung-Box no demuestra independencia completa ni ausencia de relaciones no lineales.
- ARCH modela la varianza condicional mediante errores pasados al cuadrado.
- GARCH incorpora errores pasados y valores anteriores de la propia varianza condicional.
- EGARCH permite respuestas asimétricas ante shocks positivos y negativos.
- HARCH complementa modelos heterogéneos que representan distintos horizontes temporales.
- Los modelos de volatilidad pronostican principalmente dispersión o varianza, no dirección.
- El capítulo presenta VAR para múltiples series y cointegración para detectar tendencias comunes entre activos.

## 2. Diagnósticos temporales aplicables a MNQ

Deben analizarse separadamente:

- precios;
- retornos;
- retornos absolutos;
- retornos cuadrados;
- rangos;
- volumen;
- volatilidad realizada;
- features técnicas;
- targets futuros.

Los diagnósticos deben incluir:

- ACF;
- PACF;
- Ljung-Box;
- ADF;
- distribución por régimen;
- estabilidad por año y fold;
- comparación entre periodos de alta y baja volatilidad.

ACF y PACF pueden orientar órdenes de modelos AR o MA, pero no validan directamente los lookbacks de 30, 60 y 90 minutos de CNN, LSTM, GRU o TCN.

Con más de un millón de barras, una dependencia estadísticamente significativa puede ser demasiado pequeña para poseer utilidad predictiva o económica.

## 3. Estacionalidad intradía

La estacionalidad temporal debe distinguirse de la estacionariedad.

El MNQ presenta patrones sistemáticos asociados a:

- minuto del día;
- apertura;
- cierre;
- cambios de sesión;
- noticias;
- volumen;
- rollover.

Antes de interpretar persistencia en volatilidad o volumen deben compararse:

1. series originales;
2. series normalizadas causalmente por minuto del día;
3. series normalizadas por régimen;
4. análisis separado por contrato o periodo.

Los parámetros de normalización deben estimarse únicamente con el train de cada fold.

## 4. AR y ARIMA como baselines

Para retornos futuros pueden evaluarse:

- predicción constante;
- retorno anterior;
- media rolling;
- AR(p);
- ARMA;
- ARIMA cuando la diferenciación esté justificada.

AIC y BIC pueden ayudar a filtrar órdenes candidatos, pero la selección final debe realizarse mediante validación temporal interna y rendimiento fuera de muestra.

Deben revisarse:

- estabilidad de coeficientes;
- residuos;
- sensibilidad al periodo de entrenamiento;
- rendimiento por régimen;
- costes computacionales;
- mejora frente a baselines simples.

La estacionariedad no implica predictibilidad y la autocorrelación no implica rentabilidad.

## 5. Modelos de volatilidad

Los candidatos principales son:

- volatilidad rolling;
- EWMA;
- ATR;
- rango realizado;
- realized volatility;
- ARCH;
- GARCH.

Como alternativas futuras:

- EGARCH;
- HARCH;
- otras distribuciones de errores con colas gruesas.

Un protocolo inicial puede:

1. modelar la media de los retornos o asumir media constante;
2. comprobar dependencia en residuos cuadrados;
3. ajustar un modelo de volatilidad;
4. generar pronósticos estrictamente fuera de muestra;
5. comparar contra baselines simples.

La ventana móvil de diez años empleada en el libro corresponde a su ejemplo diario del Nasdaq y no constituye una recomendación para MNQ de un minuto.

## 6. Evaluación de pronósticos de volatilidad

La varianza latente no se observa directamente.

Posibles proxies de evaluación:

- retorno futuro al cuadrado;
- suma de retornos futuros al cuadrado;
- realized volatility del horizonte;
- rango futuro;
- ATR futuro;
- excursión futura.

El retorno futuro al cuadrado es un proxy ruidoso y no debe tratarse como la varianza verdadera.

Los modelos deben evaluarse mediante:

- RMSE o MAE del proxy;
- correlación con volatilidad futura;
- estabilidad entre folds;
- utilidad incremental como feature;
- efecto sobre decisiones operativas;
- rendimiento neto cuando se utilicen como filtro.

## 7. Aplicaciones propuestas al MNQ

La volatilidad prevista puede evaluarse como:

- feature adicional;
- indicador de régimen;
- filtro para operar o no operar;
- variable para normalizar retornos;
- contexto para interpretar probabilidades;
- variable futura de gestión de riesgo.

Debe compararse si aporta información adicional respecto de:

- ATR;
- volatilidad rolling;
- EWMA;
- rangos;
- volumen relativo;
- régimen horario.

Una mejora en la predicción de volatilidad no garantiza mejora en DIR, BAR u OPC.

## 8. Barreras y position sizing

Usar volatilidad pronosticada para modificar TP, SL o tamaño de posición es una propuesta experimental futura.

Modificar dinámicamente TP y SL cambia la definición del target operativo. Por tanto, requeriría:

- definir reglas causales;
- generar targets nuevos;
- volver a etiquetar los datos;
- reentrenar los modelos;
- evaluar nuevamente el backtest.

No debe aplicarse directamente al OPC actual.

## 9. Rollover y contratos

Debe diferenciarse:

- serie destinada a features;
- precios reales de cada contrato;
- serie utilizada para ejecución y P&L.

Los diagnósticos de estacionariedad y autocorrelación deben comprobar si el procedimiento de consolidación introduce:

- saltos;
- retornos artificiales;
- cambios de escala;
- patrones alrededor del rollover.

Un ajuste retrospectivo no garantiza automáticamente una serie apropiada para todos los análisis.

## 10. Contenido de baja prioridad

### SARIMAX

Puede incorporar estacionalidad y variables exógenas, pero los ejemplos macroeconómicos mensuales del capítulo no se trasladan directamente al MNQ de un minuto.

### VAR

Es útil para varias series relacionadas. Actualmente tiene baja prioridad porque el dataset principal contiene un único instrumento, aunque podría resultar relevante al incorporar índices, volatilidad, tasas u otros mercados.

### Cointegración y pairs trading

Son contenidos explícitos del capítulo, pero no aplicables al proyecto actual de un único instrumento.

No deben tratarse contratos consecutivos del MNQ como un par cointegrado sin:

- datos simultáneos;
- hipótesis económica;
- tratamiento de liquidez;
- reglas específicas de spread y rollover.

## 11. Decisiones pendientes

- Determinar qué series necesitan transformación.
- Definir la normalización intradía causal.
- Elegir órdenes ARIMA mediante validación interna.
- Definir ventanas de entrenamiento para modelos temporales.
- Comparar GARCH con baselines simples.
- Seleccionar el proxy de volatilidad futura.
- Evaluar estabilidad por año, fold y régimen.
- Medir el aporte incremental de la volatilidad pronosticada.
- Verificar efectos artificiales del rollover.
- Determinar si ARIMA o GARCH justifican su coste frente a features rolling sencillas.

## 12. Riesgos metodológicos

- ajustar transformaciones usando toda la serie;
- usar validación externa para elegir órdenes;
- confundir estacionalidad con clustering de volatilidad;
- tratar el ADF como prueba definitiva de estacionariedad;
- interpretar Ljung-Box como prueba de ausencia total de señal;
- seleccionar modelos solo por AIC o BIC;
- usar ACF/PACF para justificar lookbacks de redes neuronales;
- ignorar cambios estructurales;
- evaluar volatilidad contra un único proxy ruidoso;
- asumir que GARCH predice dirección;
- cambiar barreras sin volver a etiquetar;
- ignorar rollover y contratos;
- confundir significancia estadística con utilidad económica.