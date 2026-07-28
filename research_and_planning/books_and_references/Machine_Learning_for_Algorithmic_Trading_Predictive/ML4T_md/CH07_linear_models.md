# Capítulo 7 — Linear Models: From Risk Factors to Return Forecasts

## 1. Conocimiento explícito del libro

- Los modelos lineales suponen que el resultado puede expresarse como una combinación lineal de las features más un término de error.
- La inferencia busca interpretar asociaciones, parámetros e incertidumbre bajo determinados supuestos estadísticos. La predicción prioriza el rendimiento sobre datos no observados.
- El Teorema de Gauss-Markov establece condiciones bajo las cuales OLS es el estimador lineal insesgado de menor varianza.
- Entre los supuestos relevantes se encuentran:
  - especificación lineal adecuada;
  - ausencia de multicolinealidad perfecta;
  - esperanza condicional del error igual a cero;
  - homocedasticidad;
  - ausencia de dependencia serial para la eficiencia e inferencia clásica.
- Estos supuestos suelen incumplirse en datos financieros.
- Ridge aplica una penalización L2 para reducir la varianza y estabilizar coeficientes.
- Lasso aplica una penalización L1 y puede llevar coeficientes a cero.
- Lasso puede realizar selección de variables, pero con features correlacionadas puede escoger arbitrariamente una de varias variables equivalentes.
- La regresión logística transforma una combinación lineal de inputs en una probabilidad para clasificación.
- El capítulo aplica principalmente clasificación binaria de movimientos de precio y regresión de retornos de múltiples acciones.
- El IC del experimento es principalmente transversal: compara diariamente predicciones y retornos futuros entre activos.

## 2. Baselines lineales aplicables a MNQ

### Regresión

Para targets continuos:

- predicción constante mediante media o mediana del train;
- OLS;
- Ridge;
- Lasso;
- Elastic Net como extensión experimental.

Métricas:

- MAE;
- RMSE;
- distribución de errores;
- estabilidad fuera de muestra;
- IC temporal únicamente cuando las predicciones y resultados sean continuos u ordenables.

### Clasificación

Para dirección o resultados operativos:

- clase mayoritaria;
- probabilidades basadas en frecuencias del train;
- regla del signo del retorno anterior;
- reglas simples de momentum o reversión;
- regresión logística binaria;
- regresión logística multinomial y one-vs-rest como candidatos para OPC.

La logística multinomial y one-vs-rest deben compararse experimentalmente. Ninguna está demostrada como superior para MNQ.

## 3. Regularización y preprocesamiento

- Ridge, Lasso y Elastic Net requieren features en escalas comparables para que la penalización sea coherente.
- El escalador debe ajustarse únicamente con el train interno de cada fold.
- Escalado, imputación, selección de features y modelo deben integrarse en un pipeline temporal.
- Variables binarias o categóricas no deben tratarse automáticamente igual que variables continuas.
- `regime_id` no debe interpretarse necesariamente como una magnitud ordinal.
- La fuerza de regularización debe seleccionarse mediante validación interna, nunca mediante los años de evaluación externa.

## 4. Interpretación y diagnóstico

- Los coeficientes estandarizados permiten comparar magnitudes dentro de un modelo, pero no representan necesariamente importancia causal.
- Su estabilidad debe evaluarse entre:
  - folds;
  - años;
  - regímenes horarios;
  - targets;
  - conjuntos de features.
- Durbin-Watson diagnostica principalmente autocorrelación de primer orden.
- Debe complementarse con ACF/PACF de residuos y evaluación temporal fuera de muestra.
- La heterocedasticidad y autocorrelación pueden invalidar errores estándar y p-valores clásicos.
- Si se busca inferencia, deben considerarse errores estándar robustos o HAC.
- Un coeficiente significativo no garantiza capacidad predictiva ni utilidad financiera.

## 5. Contenido no transferible directamente

- CAPM, Fama-French y Fama-MacBeth se utilizan principalmente para explicar exposiciones y retornos de universos multiactivo.
- Los factores externos existen, pero no forman parte del OHLCV actual y su utilidad para MNQ de un minuto sería indirecta.
- El IC transversal del libro no equivale al IC temporal de una única serie.
- `MultipleTimeSeriesCV` aporta principios de orden temporal y evaluación repetida, pero debe adaptarse a los splits, sesiones y horizontes del proyecto.

## 6. Propuestas experimentales

- Comparar regresión logística multinomial y one-vs-rest para OPC.
- Evaluar calibración de probabilidades mediante datos internos u observaciones completamente fuera de muestra.
- Comparar Ridge, Lasso y Elastic Net para estudiar estabilidad frente a features correlacionadas.
- Entrenar modelos globales con régimen como contexto y compararlos con modelos separados por régimen.
- Utilizar residuos históricos como features solamente cuando:
  - ya sean conocidos en el instante de predicción;
  - se generen causalmente;
  - provengan de predicciones out-of-fold o de modelos entrenados exclusivamente con el pasado.
- Evaluar interacciones explícitas o transformaciones no lineales simples antes de concluir que un modelo lineal es insuficiente.

## 7. Riesgos metodológicos

- seleccionar regularización o umbrales usando evaluación externa;
- ajustar escaladores con datos futuros;
- interpretar p-valores como evidencia de rentabilidad;
- usar Lasso como único método de selección;
- comparar coeficientes de variables con escalas diferentes;
- tratar probabilidades logísticas como calibradas sin medirlas;
- utilizar residuos del target actual, todavía desconocido, como feature;
- aplicar IC temporal sin controlar dependencia serial y labels solapados;
- descartar los modelos lineales solo porque modelos complejos obtienen mejor rendimiento in-sample.