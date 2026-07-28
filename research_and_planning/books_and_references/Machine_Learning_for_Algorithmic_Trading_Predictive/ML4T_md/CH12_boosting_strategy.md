# Capítulo 12 — Boosting Your Trading Strategy

## 1. Conocimiento explícito del libro

- Boosting construye ensambles secuenciales en los que cada nuevo estimador intenta mejorar el modelo acumulado.
- Estos algoritmos son especialmente competitivos en datos tabulares estructurados.
- AdaBoost y Gradient Boosting utilizan mecanismos diferentes:
  - AdaBoost repondera observaciones según los errores acumulados.
  - Gradient Boosting ajusta nuevos modelos a los gradientes negativos o pseudorresiduos de una función de pérdida.
- Los modelos base no tienen que ser siempre decision stumps; pueden ser árboles con profundidad y número de hojas configurables.
- El learning rate reduce la contribución de cada árbol y debe ajustarse conjuntamente con el número de iteraciones.
- Una tasa de aprendizaje menor suele requerir más árboles.
- XGBoost, LightGBM y CatBoost incorporan optimizaciones de memoria, cómputo, regularización y tratamiento de datos tabulares.
- Boosting puede utilizarse para regresión, clasificación binaria, clasificación multiclase y ranking.

## 2. XGBoost

El capítulo destaca:

- uso de gradientes y derivadas de segundo orden;
- aproximación de la pérdida mediante gradientes y Hessianos;
- búsqueda aproximada de puntos de corte;
- bloques de columnas comprimidos;
- procesamiento paralelo;
- manejo de valores faltantes y datos dispersos;
- subsampling de filas y columnas;
- regularización L1 y L2;
- restricciones monótonas.

Las restricciones monótonas solo deben utilizarse cuando exista una hipótesis económica o técnica suficientemente fuerte. No deben imponerse para forzar relaciones que los datos no sostienen.

## 3. LightGBM

LightGBM introduce:

- histogramas para acelerar la búsqueda de divisiones;
- Gradient-based One-Side Sampling, GOSS;
- Exclusive Feature Bundling, EFB;
- crecimiento leaf-wise;
- tratamiento nativo de variables categóricas;
- entrenamiento en CPU o GPU;
- soporte para DART.

### GOSS

GOSS conserva prioritariamente observaciones con gradientes elevados y utiliza una fracción de las observaciones con gradientes pequeños para estimar la ganancia de los splits de forma más eficiente.

### EFB

EFB combina features dispersas o mutuamente excluyentes para reducir la dimensionalidad efectiva.

Su utilidad puede ser limitada en el dataset MNQ actual si las features son densas y pocas variables son mutuamente excluyentes.

### Crecimiento leaf-wise

LightGBM divide la hoja que ofrece la mayor ganancia, incluso cuando esto produce árboles desbalanceados.

Ventajas:

- convergencia rápida;
- alta capacidad para capturar interacciones;
- eficiencia en grandes datasets.

Riesgos:

- hojas con pocas observaciones;
- árboles excesivamente profundos;
- memorización de ruido;
- sensibilidad a cambios de régimen.

## 4. CatBoost y variables categóricas

CatBoost procesa variables categóricas sin requerir necesariamente one-hot encoding.

Puede:

- combinar categorías;
- transformar categorías en estadísticas numéricas relacionadas con el target;
- utilizar priors y suavizado;
- calcular estas estadísticas acumulativamente para reducir leakage.

LightGBM también procesa categóricas de forma nativa, pero utiliza un procedimiento diferente basado en la agrupación de niveles y la ganancia de los splits.

Para MNQ deben evaluarse cuidadosamente:

- `regime_id`;
- minuto del día;
- contrato;
- trimestre;
- año.

Riesgos:

- `regime_id` no representa necesariamente una escala ordinal;
- el contrato puede actuar como proxy de fecha;
- el año puede facilitar memorización de regímenes históricos;
- el minuto del día puede dominar el modelo mediante estacionalidad;
- cualquier codificación basada en el target debe ajustarse únicamente dentro del train.

## 5. DART

DART aplica dropout sobre árboles completos del ensamble.

Busca evitar que:

- algunos árboles dominen excesivamente la predicción;
- los árboles agregados al final se especialicen en pocas observaciones;
- el resultado dependa demasiado de un subconjunto reducido del ensamble.

Debe conservarse como variante experimental, no como configuración principal inicial.

## 6. Validación temporal

Para cada fold walk-forward debe distinguirse:

1. train principal;
2. validación interna temporal;
3. evaluación externa;
4. eventual test final intacto.

La validación interna se utiliza para:

- elegir learning rate;
- seleccionar número de árboles;
- early stopping;
- ajustar profundidad y hojas;
- seleccionar regularización;
- comparar configuraciones.

La evaluación externa no debe utilizarse reiteradamente para modificar:

- hiperparámetros;
- features;
- umbrales;
- semillas;
- reglas operativas.

Reutilizar excesivamente la validación interna también puede producir sobreajuste, aunque no contamine automáticamente la evaluación externa.

## 7. Early stopping

Early stopping selecciona el número de iteraciones cuando la métrica de validación interna deja de mejorar.

Debe cumplir:

- validación cronológicamente posterior al train;
- ausencia de ajuste con el fold externo;
- métrica definida antes del experimento;
- registro de la iteración seleccionada;
- paciencia y tolerancia documentadas.

El learning rate y el número de iteraciones forman un sistema conjunto y no deben tunearse aisladamente.

## 8. Hiperparámetros relevantes

### Comunes

- número de árboles;
- learning rate;
- profundidad;
- número de hojas;
- tamaño mínimo de hojas;
- fracción de filas;
- fracción de columnas;
- regularización L1;
- regularización L2;
- ganancia mínima para dividir;
- semillas aleatorias.

### LightGBM

- `num_leaves`;
- `max_depth`;
- `min_data_in_leaf`;
- `min_gain_to_split`;
- `feature_fraction`;
- `bagging_fraction`;
- `bagging_freq`;
- `lambda_l1`;
- `lambda_l2`.

`num_leaves < 2^max_depth` es una guía de control, no una garantía contra el overfitting.

### XGBoost

- `max_depth`;
- `min_child_weight`;
- `gamma`;
- `subsample`;
- `colsample_bytree`;
- `reg_alpha`;
- `reg_lambda`.

### CatBoost

- `depth`;
- `learning_rate`;
- `iterations`;
- `l2_leaf_reg`;
- parámetros de muestreo;
- tratamiento de variables categóricas.

## 9. Aplicación tabular al MNQ

Los modelos boosting reciben una matriz tabular.

Las ventanas de 30, 60 y 90 minutos pueden representarse mediante:

- retornos rezagados;
- estadísticas rolling;
- rangos;
- volatilidad;
- momentum;
- ATR;
- RSI;
- volumen relativo;
- posición en rango;
- variables temporales;
- agregaciones causales.

No procesan la secuencia del mismo modo que CNN1D, LSTM, GRU o TCN.

Deben compararse:

- representación compacta;
- representación con niveles de precio;
- representación sin niveles;
- lags explícitos;
- estadísticas agregadas.

## 10. Targets aplicables

### Clasificación

LightGBM, XGBoost y CatBoost pueden evaluarse para:

- DIR;
- BAR;
- OPC multiclase.

Métricas:

- macro F1;
- balanced accuracy;
- log loss;
- métricas por clase;
- matriz de confusión;
- estabilidad por fold.

### Regresión

Pueden evaluarse para:

- retornos futuros;
- magnitud de movimiento;
- excursiones;
- volatilidad futura.

El horizonte debe coincidir con el target real de 30, 60 o 90 minutos.

Métricas:

- MAE;
- RMSE;
- correlación fuera de muestra;
- IC temporal cuando sea metodológicamente apropiado.

## 11. Probabilidades y calibración

Los scores de clasificación de boosting no deben asumirse calibrados.

Deben analizarse mediante:

- log loss;
- Brier Score multiclase;
- curvas de calibración;
- métricas por nivel de confianza;
- calibración por clase;
- estabilidad entre años y folds.

Cualquier calibrador debe ajustarse exclusivamente dentro del train.

Las probabilidades no deben utilizarse para position sizing sin una validación separada.

## 12. SHAP e interpretación

SHAP descompone la salida del modelo respecto de un valor base mediante contribuciones aditivas de las features.

Puede utilizarse para:

- importancia global;
- explicación local;
- análisis por clase;
- interacciones;
- comparación por fold;
- comparación por régimen.

SHAP no demuestra:

- causalidad;
- estabilidad económica;
- utilidad financiera;
- que una feature deba conservarse;
- que una asociación persistirá en vivo.

Las features correlacionadas pueden compartir o redistribuir atribuciones.

Para selección de features, SHAP debe calcularse sobre:

- validación interna;
- predicciones OOF generadas dentro del train;
- folds internos no usados para ajustar el modelo analizado.

El SHAP de evaluación externa debe limitarse a diagnóstico final, sin rediseñar repetidamente el modelo.

## 13. Ejemplo intradía del capítulo

El capítulo utiliza:

- acciones del Nasdaq-100;
- barras de un minuto de AlgoSeek;
- aproximadamente 51 millones de observaciones;
- datos de trades y quotes;
- operaciones en bid y ask;
- upticks y downticks;
- indicadores técnicos;
- retornos rezagados.

El objetivo es predecir el retorno futuro a un minuto calculado sobre un precio medio ponderado por volumen.

El modelo entrena con aproximadamente doce meses y predice los siguientes veintiún días de mercado mediante múltiples splits temporales.

Este ejemplo demuestra que LightGBM puede escalar a datos intradía, pero no valida directamente el proyecto MNQ porque:

- utiliza múltiples acciones;
- posee información de microestructura;
- utiliza un horizonte de un minuto;
- opera durante el horario regular de acciones;
- el target y la estrategia son diferentes;
- MNQ dispone actualmente solo de OHLCV.

## 14. Subsampling y coste computacional

El dataset MNQ contiene más de un millón de filas, por lo que el coste computacional debe medirse.

El subsampling experimental debe conservar:

- todos los años;
- regímenes horarios;
- contratos;
- clases;
- sesiones de alta y baja volatilidad.

Alternativas:

- sesiones completas;
- bloques cronológicos;
- fracciones estratificadas dentro del train;
- reducción inicial del espacio de hiperparámetros;
- entrenamiento con histogramas;
- paralelización;
- GPU cuando proporcione una ventaja real.

Una muestra aleatoria sin control puede eliminar periodos poco frecuentes pero importantes.

## 15. Comparaciones necesarias

Boosting debe compararse con:

- predicción constante;
- reglas heurísticas;
- regresión logística;
- árbol individual;
- Random Forest;
- HistGradientBoosting;
- MLP;
- modelos secuenciales posteriores.

La comparación debe separar:

### Rendimiento predictivo

- macro F1;
- balanced accuracy;
- log loss;
- MAE o RMSE;
- estabilidad por fold.

### Calibración

- Brier;
- curvas de calibración;
- confianza por clase.

### Interpretabilidad

- SHAP;
- permutation importance;
- estabilidad de importancias.

### Rendimiento financiero

- P&L bruto;
- P&L neto;
- costes;
- turnover;
- drawdown;
- estabilidad por año y régimen.

## 16. Contenido no transferible directamente

No son directamente aplicables:

- ranking transversal de acciones;
- deciles o quintiles entre activos;
- posiciones long-short market-neutral;
- neutralización sectorial;
- optimización de un universo de acciones.

El MNQ es actualmente un único instrumento y requiere decisiones temporales, no rankings cross-sectional.

## 17. Decisiones pendientes

- Elegir qué implementación de boosting evaluar primero.
- Definir un espacio de hiperparámetros limitado.
- Diseñar la validación interna temporal.
- Definir el tratamiento de categóricas.
- Comparar representación global y modelos por régimen.
- Evaluar coste CPU frente a GPU.
- Diseñar calibración OOF.
- Definir protocolo SHAP sin contaminar evaluación.
- Comparar contra Random Forest y regresión logística.
- Medir estabilidad por clase, fold, año y régimen.
- Verificar si la complejidad aporta utilidad incremental real.

## 18. Riesgos metodológicos

- usar evaluación externa para tuning;
- reutilizar excesivamente validación interna;
- árboles leaf-wise demasiado complejos;
- hojas con muy pocas observaciones;
- memorizar año, contrato o minuto;
- target encoding con información futura;
- interpretar SHAP como causalidad;
- seleccionar features usando SHAP in-sample;
- asumir probabilidades calibradas;
- confundir el ejemplo intradía de acciones con MNQ;
- reducir datos sin conservar cobertura temporal;
- seleccionar el modelo únicamente por P&L;
- asumir que boosting siempre supera a modelos simples.