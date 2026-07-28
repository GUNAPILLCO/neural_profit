# Capítulo 8 — The ML4T Workflow: From Model to Strategy Backtesting

## 1. Conocimiento explícito del libro

- El backtesting busca reunir evidencia para rechazar o conservar provisionalmente una hipótesis de inversión, no demostrar definitivamente que una estrategia funcionará.
- Entre sus principales riesgos se encuentran:
  - look-ahead bias;
  - survivorship bias;
  - data snooping;
  - errores en datos, ejecución y supuestos estadísticos.
- Un backtest vectorizado permite evaluar rápidamente estrategias simples mediante señales y retornos correctamente alineados.
- Un motor basado en eventos facilita la simulación de calendarios, órdenes, posiciones, capital y reglas dependientes del estado.
- Ningún enfoque elimina automáticamente los sesgos si los datos, timestamps o reglas son incorrectos.
- El Deflated Sharpe Ratio ajusta la evaluación del Sharpe por múltiples pruebas, longitud de muestra y características no normales de los retornos.
- El capítulo utiliza backtrader y Zipline como ejemplos de arquitectura; sus principios metodológicos son más importantes que sus versiones concretas.

## 2. Arquitectura aplicable a MNQ

El proceso debe distinguir:

1. **Backtest preliminar**
   - predicciones fuera de muestra;
   - señales correctamente desplazadas;
   - costes básicos;
   - reglas simples y previamente definidas.

2. **Desarrollo de reglas**
   - conversión de probabilidades o scores en posiciones;
   - TP, SL y duración;
   - filtros;
   - tamaño de posición;
   - tratamiento de señales consecutivas.

3. **Evaluación walk-forward externa**
   - predicciones estrictamente out-of-sample;
   - frecuencia de reentrenamiento coherente con el sistema futuro;
   - reglas congeladas para cada evaluación.

4. **Test final**
   - periodo no utilizado para modificar datos, features, targets, modelos o ejecución;
   - evaluación única después de cerrar el protocolo.

Los periodos externos dejan de ser completamente independientes cuando sus resultados se utilizan repetidamente para modificar la estrategia.

## 3. Causalidad y ejecución

- Debe verificarse si el timestamp representa el inicio o el final de la barra.
- Si una señal necesita el `close`, `high`, `low` o volumen completo de la barra `t`, solo existe después de finalizar esa barra.
- La primera ejecución causal será un evento posterior, normalmente en la barra siguiente, sujeto a disponibilidad de precio y slippage.
- No debe asumirse una entrada al mismo `close_t` utilizado para generar la señal.
- Debe auditarse si los targets DIR, BAR u OPC utilizan como entrada un precio que sería realmente ejecutable.
- OHLCV no permite conocer la secuencia intraminuto ni el precio exacto de ejecución.

Cuando TP y SL se alcanzan en la misma vela deben compararse:

- peor caso;
- mejor caso;
- exclusión;
- clase ambigua;
- análisis de sensibilidad.

También debe medirse la frecuencia de estos casos.

## 4. Backtest vectorizado y basado en eventos

Un backtest vectorizado puede ser suficiente cuando:

- las reglas son simples;
- la señal determina directamente la posición del periodo siguiente;
- no existen órdenes pendientes;
- los costes pueden representarse razonablemente;
- no hay lógica intrabarra compleja.

Un motor basado en eventos es preferible cuando existen:

- órdenes límite, stop o take profit;
- posiciones y órdenes pendientes;
- restricciones de capital;
- cancelaciones;
- rollover;
- reglas dependientes del estado.

Sin ticks ni libro de órdenes, un motor basado en eventos no puede reconstruir:

- secuencia intraminuto;
- spread real;
- liquidez disponible;
- fills parciales;
- impacto de mercado.

## 5. Predicciones precalculadas

El entrenamiento puede desacoplarse del backtest si las predicciones:

- fueron generadas estrictamente fuera de muestra;
- usan solamente información disponible hasta cada instante;
- reproducen la frecuencia prevista de reentrenamiento;
- conservan versión del modelo, features, parámetros y fecha de entrenamiento;
- no fueron regeneradas después de observar el resultado financiero externo.

## 6. Costes de operación

Deben modelarse por separado:

- comisiones del broker;
- tasas del exchange y regulatorias;
- spread;
- slippage;
- impacto de mercado;
- coste por lado;
- coste de ida y vuelta;
- costes del rollover.

Los escenarios de 0, 1 o 2 ticks son análisis de sensibilidad, no estimaciones confirmadas del coste real.

`(O + H + L + C) / 4` puede utilizarse como feature, pero no como VWAP real ni como precio necesariamente ejecutable.

## 7. Tratamiento del rollover

Debe distinguirse:

- serie ajustada o normalizada utilizada para features;
- precio real del contrato operado para órdenes y P&L.

La simulación debe documentar:

- contrato activo;
- fecha o criterio de cambio;
- cierre del contrato anterior;
- apertura del nuevo;
- diferencia de precios;
- comisiones y slippage del roll.

## 8. Evaluación estadística y múltiples pruebas

- El Sharpe observado aumenta artificialmente cuando se prueban numerosas estrategias y se selecciona la mejor.
- El DSR puede complementar la evaluación, pero requiere estimar correctamente:
  - número efectivo de configuraciones probadas;
  - dependencia entre experimentos;
  - asimetría y curtosis;
  - longitud efectiva de la muestra.
- Los ejemplos del libro sobre cantidad de pruebas soportadas por dos o cinco años son ilustrativos y no se trasladan literalmente a barras de un minuto.
- La cantidad relevante es el número efectivo de observaciones o apuestas independientes.
- El DSR no reemplaza:
  - walk-forward;
  - test final;
  - costes;
  - análisis por periodos;
  - estabilidad de resultados.

## 9. Decisiones pendientes

- Verificar semántica temporal del dataset.
- Auditar el precio de entrada utilizado por cada target.
- Definir reglas para velas TP/SL ambiguas.
- Estimar costes reales y escenarios de sensibilidad.
- Elegir cuándo basta un backtest vectorizado y cuándo se requiere uno por eventos.
- Definir reglas de rollover.
- Determinar qué periodo todavía puede reservarse como test final.
- Registrar el número de configuraciones, targets, features y reglas probadas.
- Comparar resultados por fold, año, régimen, dirección y clase.

## 10. Riesgos metodológicos

- predicciones in-sample;
- ejecutar al mismo cierre utilizado para calcular la señal;
- optimizar reglas con la evaluación externa;
- asumir fills exactos en TP, SL o límites;
- ignorar casos intrabarra ambiguos;
- usar costes arbitrarios como si fueran reales;
- calcular P&L con precios continuos ajustados;
- considerar cada minuto como una observación independiente;
- confiar en un motor por eventos sin datos suficientes;
- confundir una mejora predictiva con utilidad financiera.