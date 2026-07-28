# Capítulo 11 — Random Forests: A Long-Short Strategy for Japanese Stocks

## 1. Conocimiento explícito del libro

- Los árboles de decisión construyen reglas binarias recursivas para dividir el espacio de las features.
- Los árboles individuales pueden representar relaciones no lineales e interacciones, pero tienden a presentar alta varianza y sobreajuste.
- En regresión, las divisiones suelen buscar una reducción del error cuadrático.
- En clasificación, se utilizan criterios como:
  - impureza de Gini;
  - entropía;
  - error de clasificación.
- Bagging entrena múltiples árboles sobre muestras bootstrap y agrega sus predicciones.
- La reducción de varianza depende de que los errores de los árboles no estén excesivamente correlacionados.
- Random Forest añade selección aleatoria de features para decorrelacionar adicionalmente los árboles.
- Reducir `max_features` puede disminuir la correlación entre árboles, aunque también puede aumentar el bias.
- El OOB score usa observaciones no seleccionadas en cada bootstrap para generar predicciones internas.
- El libro advierte que el muestreo fuera de orden hace muy difícil utilizar OOB correctamente con series temporales.
- Los bosques son menos interpretables que un árbol individual y pueden requerir mucha memoria, entrenamiento y tiempo de inferencia.
- En los experimentos del capítulo, modelos más complejos no superaron necesariamente a árboles individuales.
- La estrategia final utiliza LightGBM en modo Random Forest para predecir retornos de acciones japonesas y construir posiciones long-short mediante rankings transversales.

## 2. Aplicación tabular al MNQ

Random Forest recibe normalmente una matriz tabular:

- una fila por instante de predicción;
- una columna por feature;
- un target asociado a cada fila.

Las ventanas históricas de 30, 60 y 90 minutos pueden representarse mediante:

- estadísticas rolling;
- retornos rezagados seleccionados;
- rangos y volatilidad;
- indicadores técnicos;
- volumen normalizado;
- máximos y mínimos relativos;
- features de contexto temporal;
- secuencias aplanadas, cuando estén justificadas.

Aplanar todos los minutos puede:

- aumentar mucho la dimensionalidad;
- generar features redundantes;
- elevar el consumo de memoria;
- perder una representación explícita del orden temporal.

Por ello, Random Forest debe considerarse un baseline tabular no lineal, distinto de CNN1D, LSTM, GRU y TCN.

## 3. Targets aplicables

### Clasificación

Random Forest puede evaluarse para:

- DIR;
- BAR;
- OPC multiclase;
- otros targets categóricos futuros.

Debe compararse contra:

- clase mayoritaria;
- probabilidades por frecuencia;
- reglas heurísticas;
- regresión logística;
- árbol individual.

### Regresión

`RandomForestRegressor` puede utilizarse para:

- retorno futuro;
- magnitud de movimiento;
- excursión;
- volatilidad futura;
- otros targets continuos.

El horizonte debe coincidir exactamente con el target evaluado, incluyendo los candidatos de 30, 60 y 90 minutos.

## 4. Hiperparámetros principales

### Complejidad del árbol

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_leaf_nodes`;
- criterio de división.

### Configuración del bosque

- `n_estimators`;
- `max_features`;
- `bootstrap`;
- `max_samples`;
- `class_weight`;
- `random_state`;
- paralelización.

Efectos esperados:

- árboles demasiado profundos pueden memorizar ruido;
- hojas pequeñas aumentan la varianza;
- `max_features` bajo puede decorrelacionar árboles, pero aumentar bias;
- más árboles suelen estabilizar el bosque, pero elevan coste y memoria;
- `class_weight` puede ayudar con desbalance, pero debe ajustarse únicamente con el train.

Los hiperparámetros deben seleccionarse mediante validación interna temporal, nunca utilizando los años de evaluación externa.

## 5. Validación temporal

La evaluación principal debe respetar:

- train anterior a validación;
- sesiones y timestamps;
- horizontes de las etiquetas;
- posibles labels solapados;
- preprocesamiento ajustado únicamente con train.

Pueden utilizarse:

- walk-forward expansivo;
- ventanas móviles;
- divisores temporales personalizados;
- `MultipleTimeSeriesCV` adaptado.

Los splits aleatorios no deben utilizarse para afirmar generalización futura del MNQ.

## 6. OOB y bootstrap

OOB es contenido explícito del libro, pero no debe ser el estimador principal para MNQ porque:

- el bootstrap ignora el orden cronológico;
- observaciones próximas pueden ser muy similares;
- los targets pueden solaparse;
- muestras OOB pueden estar temporalmente rodeadas por muestras utilizadas para entrenar.

Clasificación:

- **A:** técnica explícita del capítulo;
- **D como validación principal para MNQ**;
- posible diagnóstico interno secundario.

Experimentos futuros posibles:

- bootstrap estándar;
- entrenamiento sin bootstrap;
- `max_samples` reducido;
- submuestreo sin reemplazo;
- muestreo por sesiones o bloques.

La evaluación definitiva debe continuar siendo temporal.

## 7. Importancia de features

### Importancia por impureza

Se calcula durante el entrenamiento a partir de la reducción acumulada del criterio de división.

Limitaciones:

- es in-sample;
- puede favorecer variables con muchos puntos posibles de corte;
- puede repartir importancia entre features correlacionadas;
- puede asignar importancia a proxies de régimen o tiempo;
- no demuestra causalidad;
- no garantiza utilidad financiera.

### Permutation importance

Debe calcularse principalmente sobre:

- validación interna;
- predicciones out-of-fold dentro del train;
- periodos no utilizados para ajustar el modelo.

La evaluación externa puede utilizarse como diagnóstico final, pero no para seleccionar repetidamente features.

Como propuesta para MNQ puede evaluarse:

- permutación por bloques temporales;
- permutación por sesiones;
- permutación conjunta de grupos correlacionados;
- comparación de importancia entre folds y años.

Permutar minutos individualmente puede destruir la estructura temporal y generar una estimación difícil de interpretar.

## 8. Probabilidades y calibración

Random Forest obtiene probabilidades agregando las estimaciones de los árboles. En cada árbol, estas dependen de la distribución de clases observada en las hojas.

Estas probabilidades no están automáticamente calibradas.

Para DIR, BAR y OPC deben analizarse:

- log loss;
- Brier Score o extensión multiclase;
- curvas de calibración;
- error de calibración;
- métricas por clase;
- estabilidad entre folds;
- distribución de confianza.

No deben utilizarse directamente para position sizing sin validación adicional.

## 9. Evaluación del modelo

### Rendimiento predictivo

- macro F1;
- balanced accuracy;
- log loss;
- matriz de confusión;
- métricas por clase;
- MAE o RMSE para regresión;
- estabilidad entre folds.

### Calibración

- Brier;
- curvas de calibración;
- calidad de probabilidades por clase.

### Rendimiento financiero

- P&L bruto y neto;
- costes;
- Sharpe;
- drawdown;
- turnover;
- rendimiento por año y régimen.

Estas dimensiones deben permanecer separadas. Un modelo puede mejorar macro F1 sin mejorar P&L, y un resultado financiero favorable puede depender de reglas operativas distintas del modelo predictivo.

## 10. Análisis de estabilidad

Random Forest debe analizarse por:

- fold;
- año;
- régimen horario;
- dirección;
- clase;
- horizonte;
- conjunto de features;
- semilla aleatoria.

También deben compararse:

- modelo global con régimen como contexto;
- modelos independientes por régimen;
- árbol individual;
- Random Forest;
- baselines lineales.

Las curvas de aprendizaje pueden ayudar a distinguir:

- falta de datos;
- exceso de varianza;
- exceso de bias;
- saturación del modelo.

## 11. Contenido no transferible directamente

La estrategia japonesa del capítulo utiliza:

- universo de múltiples acciones;
- ranking transversal;
- quintiles;
- posiciones long-short;
- neutralidad de mercado;
- variables sectoriales;
- evaluación relativa entre activos.

Estos elementos no se trasladan al proyecto actual de un único futuro MNQ.

Los indicadores técnicos utilizados por el capítulo son features candidatas. El libro no demuestra que sean predictivos para MNQ de un minuto.

## 12. Decisiones pendientes

- Definir la representación tabular de las ventanas.
- Establecer un espacio de hiperparámetros razonable.
- Determinar si conviene bootstrap en datos intradía.
- Definir el tratamiento del desbalance.
- Comparar modelo global y modelos por régimen.
- Evaluar calibración de probabilidades.
- Diseñar permutation importance temporal.
- Medir estabilidad de importancias.
- Comparar RF con regresión logística y árbol individual.
- Estimar coste computacional con más de un millón de observaciones.
- Definir qué resultados externos podrán utilizarse solo como diagnóstico final.

## 13. Riesgos metodológicos

- tuning con evaluación externa;
- splits aleatorios;
- confiar en OOB como validación temporal;
- árboles excesivamente profundos;
- hojas demasiado pequeñas;
- alta redundancia entre observaciones;
- interpretar importancia como causalidad;
- seleccionar features con un único fold;
- permutar minutos sin respetar dependencia;
- asumir probabilidades calibradas;
- ocultar clases minoritarias con métricas agregadas;
- confundir las ventanas tabulares con secuencias neuronales;
- asumir que un ensamble siempre supera a modelos simples;
- seleccionar el modelo únicamente por P&L.