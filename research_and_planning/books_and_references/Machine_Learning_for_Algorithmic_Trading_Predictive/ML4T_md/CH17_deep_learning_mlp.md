# Capítulo 17 — Deep Learning for Trading

## 1. Conocimiento explícito del libro

El capítulo introduce las redes neuronales feedforward como base conceptual de las arquitecturas de Deep Learning estudiadas posteriormente.

Una red feedforward está compuesta por:

- capa de entrada;
- una o más capas ocultas;
- capa de salida;
- conexiones ponderadas;
- funciones de activación no lineales.

El forward pass transforma los inputs en una predicción.

Backpropagation aplica la regla de la cadena para calcular el gradiente de la función de pérdida respecto de los parámetros. Un optimizador utiliza estos gradientes para actualizar pesos y sesgos.

Deep Learning puede aprender representaciones jerárquicas útiles para una tarea. Esto puede reducir la dependencia de features manuales, pero no elimina la necesidad de:

- construir inputs causales;
- seleccionar un target correcto;
- tratar datos faltantes;
- controlar la escala;
- evitar leakage;
- validar fuera de muestra.

## 2. Capacidad de aproximación

El teorema de aproximación universal establece, bajo determinadas condiciones, que una red con suficiente capacidad puede aproximar funciones continuas.

No garantiza que:

- el optimizador encuentre esa función;
- la red generalice;
- existan suficientes datos;
- el problema sea económicamente predecible;
- una arquitectura profunda sea mejor;
- el entrenamiento sea estable o eficiente.

Una capacidad elevada también incrementa el riesgo de memorizar ruido financiero.

## 3. MLP como baseline neuronal para MNQ

El MLP será un baseline neuronal tabular.

Su entrada debe tener dimensión fija. Puede contener:

- features base actuales;
- estadísticas rolling;
- retornos rezagados seleccionados;
- indicadores técnicos;
- variables de volatilidad;
- volumen relativo;
- contexto temporal;
- secuencias aplanadas;
- representaciones reducidas.

No es obligatorio incluir todos los lags de 30, 60 o 90 minutos.

Aunque el nombre de cada columna puede codificar su posición temporal, el MLP no incorpora explícitamente:

- convoluciones locales;
- recurrencia;
- memoria;
- parámetros compartidos a través del tiempo;
- invariancia respecto de la posición.

Por tanto, no equivale a CNN1D, LSTM, GRU o TCN.

## 4. Activaciones, outputs y pérdidas

### Regresión

Para retornos, magnitud o volatilidad futura:

- salida lineal;
- MSE como baseline;
- MAE o Huber como alternativas robustas.

MSE penaliza fuertemente errores grandes y puede ser sensible a valores extremos.

### Clasificación binaria

Para un target binario como DIR:

- una unidad de salida;
- activación sigmoide;
- binary cross-entropy.

### Clasificación multiclase

Para OPC u otro target de clases mutuamente excluyentes:

- una unidad por clase;
- activación softmax;
- sparse categorical cross-entropy o categorical cross-entropy, según la codificación.

BAR deberá configurarse según sea binario, multiclase o continuo en cada experimento.

Las probabilidades sigmoide o softmax no están automáticamente calibradas.

## 5. Activaciones ocultas

El capítulo analiza activaciones como:

- sigmoide;
- tanh;
- ReLU.

Sigmoide y tanh pueden saturarse, reduciendo la magnitud de los gradientes.

ReLU facilita el entrenamiento en muchas arquitecturas, pero puede:

- generar unidades muertas;
- producir activaciones sin límite superior;
- seguir siendo sensible al learning rate y a la inicialización.

También pueden considerarse experimentalmente variantes como Leaky ReLU, pero no son necesarias para el baseline inicial.

## 6. Inicialización

La inicialización influye en:

- propagación de activaciones;
- magnitud de gradientes;
- velocidad de convergencia;
- estabilidad entre semillas.

Como principio general:

- inicialización He es apropiada para ReLU;
- Glorot/Xavier suele asociarse con tanh o sigmoide.

La inicialización debe registrarse junto con la semilla y el resto de la configuración.

## 7. Optimización

Optimizadores candidatos:

- SGD con momentum;
- Adam;
- AdamW;
- RMSprop como alternativa secundaria.

Adam combina estimaciones adaptativas de primer y segundo momento, pero no está garantizado que generalice mejor que SGD.

El resultado depende conjuntamente de:

- learning rate;
- batch size;
- arquitectura;
- inicialización;
- regularización;
- número de épocas;
- optimizador.

No deben tunearse como decisiones completamente independientes.

## 8. Regularización

Las técnicas principales son:

- early stopping;
- L2;
- L1;
- dropout;
- reducción de profundidad o ancho.

No deben acumularse automáticamente.

Protocolo inicial sugerido:

1. arquitectura pequeña con early stopping;
2. añadir L2 si existe overfitting;
3. evaluar dropout como alternativa;
4. combinar L2 y dropout solo si aportan valor incremental.

Un exceso de regularización puede producir underfitting.

## 9. Early stopping

Early stopping debe utilizar una validación interna temporal dentro del train de cada fold.

Debe registrar:

- métrica monitorizada;
- paciencia;
- mejora mínima;
- época seleccionada;
- mejor estado del modelo;
- máximo de épocas.

La evaluación walk-forward externa no debe utilizarse para:

- elegir la época;
- seleccionar arquitectura;
- ajustar learning rate;
- decidir dropout;
- comparar semillas.

Consultar repetidamente la misma validación interna para muchas decisiones también puede sobreajustarla.

## 10. Batch size y shuffling

El batch size afecta:

- ruido del gradiente;
- número de actualizaciones;
- estabilidad;
- velocidad;
- uso de memoria;
- posible generalización.

Debe seleccionarse mediante rendimiento predictivo en validación interna, no directamente mediante el P&L externo.

Para un MLP tabular no stateful puede mezclarse el orden de las filas dentro del train cuando:

- las features sean causales;
- las etiquetas estén correctamente construidas;
- train y validación permanezcan separados;
- ninguna ventana atraviese sesiones o splits prohibidos;
- el preprocesamiento se haya ajustado solo con train.

El shuffling interno no convierte en aleatoria la evaluación temporal.

## 11. Pipeline causal

Dentro de cada fold deben ajustarse utilizando únicamente el train:

- imputación;
- tratamiento de outliers;
- escalado;
- selección de features;
- PCA o transformaciones;
- balanceo;
- pesos de clase;
- calibradores.

La validación interna y la evaluación externa solo deben transformarse mediante los parámetros ya aprendidos.

No es obligatorio que todo se implemente mediante `sklearn.Pipeline`, pero debe existir un flujo reproducible equivalente.

## 12. Aplicación a los targets

Deben establecerse inicialmente modelos separados:

- MLP para DIR;
- MLP para BAR;
- MLP para OPC;
- MLP de regresión para targets continuos.

La red multitarea queda como experimento posterior.

Antes de aplicarla deberán definirse:

- targets compatibles;
- horizontes;
- cabezas de salida;
- pesos de cada pérdida;
- desbalance;
- posibles conflictos entre gradientes.

Compartir capas no garantiza que ambas tareas se beneficien.

## 13. Métricas

### Rendimiento predictivo

Clasificación:

- macro F1;
- balanced accuracy;
- log loss;
- matriz de confusión;
- métricas por clase;
- ROC-AUC o PR-AUC cuando estén metodológicamente definidas.

Regresión:

- MAE;
- RMSE;
- correlación fuera de muestra;
- IC temporal cuando corresponda.

### Calibración

- Brier binario o multiclase;
- curvas de calibración;
- error de calibración;
- confianza por clase.

### Rendimiento financiero

- P&L bruto;
- P&L neto;
- costes;
- turnover;
- Sharpe;
- drawdown;
- estabilidad por año y régimen.

Estas dimensiones deben reportarse por separado.

## 14. Estabilidad entre semillas

El rendimiento debe analizarse utilizando un conjunto de semillas definido previamente.

Debe reportarse:

- media;
- mediana;
- desviación estándar;
- mínimo y máximo;
- resultados por fold;
- estabilidad de predicciones;
- estabilidad financiera.

No deben probarse semillas hasta encontrar una favorable.

Un ensemble de semillas constituye una estrategia diferente y debe evaluarse explícitamente.

## 15. Ensemble de modelos

El capítulo selecciona arquitecturas mediante validación temporal y combina las señales de varios modelos antes de evaluarlas en un periodo final.

Para MNQ, un ensemble puede evaluarse posteriormente mediante:

- promedio de probabilidades;
- promedio de predicciones continuas;
- promedio de semillas;
- combinación de arquitecturas.

Toda selección y ponderación debe ocurrir dentro del train.

Alternativas más controladas:

- promedio de un conjunto predefinido;
- top-k seleccionado mediante validación interna;
- pesos calculados con predicciones OOF;
- ensemble uniforme para reducir grados de libertad.

El fold externo no debe utilizarse para elegir integrantes o pesos.

## 16. Comparaciones necesarias

El MLP debe competir con:

- modelo constante;
- reglas heurísticas;
- regresión logística;
- Random Forest;
- LightGBM;
- XGBoost;
- CatBoost.

Debe comprobarse si aporta:

- mejor rendimiento predictivo;
- mejor calibración;
- mayor estabilidad;
- utilidad financiera incremental;
- una mejora suficiente para justificar su coste.

En datos tabulares, boosting puede superar al MLP.

## 17. Ejemplo del capítulo

El ejemplo utiliza:

- 995 acciones estadounidenses;
- datos diarios entre 2010 y 2017;
- volatilidad;
- momentum;
- retornos rezagados;
- rankings transversales;
- rankings sectoriales;
- validación temporal;
- ensemble de modelos;
- estrategia long-short.

Son transferibles:

- construcción flexible de MLP;
- comparación de profundidad y ancho;
- dropout;
- batch size;
- optimización;
- validación temporal;
- análisis de sensibilidad;
- ensemble como concepto.

No son directamente transferibles:

- ranking entre acciones;
- ranking sectorial;
- IC transversal diario;
- long-short market-neutral;
- frecuencia diaria;
- selección concreta de modelos del experimento.

## 18. Frameworks y recursos

El capítulo utiliza TensorFlow 2, Keras y PyTorch.

También presenta TensorBoard para:

- comparar ejecuciones;
- observar pérdidas;
- inspeccionar pesos;
- visualizar el grafo computacional;
- diagnosticar entrenamiento.

Las versiones del libro son antiguas. Deben conservarse los principios, no copiar literalmente la sintaxis.

La GPU puede acelerar operaciones matriciales, pero:

- no mejora automáticamente la generalización;
- no corrige leakage;
- no evita overfitting;
- puede no aportar ventaja en redes pequeñas;
- no garantiza reproducibilidad exacta.

## 19. Decisiones pendientes

- Definir la representación tabular oficial del MLP.
- Elegir una arquitectura inicial pequeña.
- Definir optimizadores candidatos.
- Establecer learning rates.
- Comparar regularización.
- Definir el protocolo de semillas.
- Diseñar la validación interna temporal.
- Evaluar class weights para OPC.
- Diseñar calibración OOF.
- Comparar contra boosting.
- Medir coste CPU y GPU.
- Definir si un ensemble se evaluará en una etapa posterior.

## 20. Riesgos metodológicos

- arquitectura demasiado grande;
- tuning con evaluación externa;
- reutilización excesiva de validación interna;
- escalado con datos futuros;
- seleccionar semillas favorables;
- interpretar softmax como calibración;
- acumular regularización sin comprobar underfitting;
- elegir batch size por P&L;
- confundir MLP tabular con modelo secuencial;
- asumir que Deep Learning supera a boosting;
- seleccionar ensemble después de observar el fold externo;
- medir éxito solo mediante error de entrenamiento;
- considerar capacidad de aproximación como evidencia de predictibilidad financiera.