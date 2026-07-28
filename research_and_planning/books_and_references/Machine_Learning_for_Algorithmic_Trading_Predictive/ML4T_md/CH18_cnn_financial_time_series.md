# Capítulo 18 — CNNs for Financial Time Series and Satellite Images

## 1. Conocimiento explícito del libro

Las redes neuronales convolucionales incorporan el supuesto de que los datos presentan una estructura de rejilla y que las relaciones locales pueden contener información útil.

En una serie temporal:

- el eje temporal constituye una rejilla unidimensional;
- observaciones cercanas pueden presentar relaciones relevantes;
- varias series o features pueden actuar como canales;
- los filtros se desplazan sobre el tiempo y comparten parámetros entre posiciones.

La utilidad de una CNN depende de que los patrones locales sean relevantes para el target. Una estructura temporal local no implica automáticamente predictibilidad financiera.

## 2. Convolución y compartición de parámetros

Una capa convolucional aplica filtros aprendidos sobre regiones locales de la entrada.

Cada filtro produce un feature map que identifica una determinada configuración local.

La compartición de parámetros implica que el mismo filtro se aplica en todas las posiciones temporales.

Ventajas:

- menor cantidad de parámetros que una capa densa comparable;
- reutilización de detectores locales;
- eficiencia computacional;
- aprendizaje jerárquico mediante varias capas.

Limitaciones:

- el mismo patrón puede tener significados distintos según el momento del día;
- la compartición puede ser inadecuada cuando la posición exacta importa;
- capas densas posteriores a `Flatten` pueden volver a elevar notablemente el número de parámetros.

Para MNQ, un patrón idéntico puede comportarse de forma diferente durante Overnight, Opening o Closing.

## 3. Kernel, stride y padding

### Kernel size

Define cuántos pasos temporales examina localmente cada filtro.

Kernels pequeños:

- buscan patrones de corta duración;
- permiten aumentar progresivamente el campo receptivo mediante profundidad;
- suelen requerir menos parámetros.

Kernels grandes:

- cubren directamente intervalos mayores;
- pueden mezclar varias dinámicas;
- elevan el riesgo de sobreajuste.

### Stride

Define cuánto se desplaza el filtro entre aplicaciones.

Un stride mayor reduce la resolución temporal y el coste computacional, pero puede omitir patrones breves.

### Padding

Opciones principales:

- `valid`: no añade relleno y reduce la longitud;
- `same`: conserva aproximadamente la longitud;
- `causal`: rellena a la izquierda para que la salida de cada posición no utilice posiciones posteriores.

El libro utiliza `padding='causal'` en su CNN1D mensual.

Para una predicción many-to-one de MNQ:

- toda la ventana termina en `t`;
- la predicción corresponde a un resultado posterior;
- `same`, `valid` o `causal` pueden ser compatibles con causalidad.

El padding causal no garantiza por sí mismo ausencia de leakage. La causalidad depende de todo el pipeline.

## 4. Pooling y agregación temporal

Pooling reduce la resolución de los feature maps.

### Max pooling

Conserva la activación máxima de cada región.

Puede ser útil para detectar la presencia de un patrón fuerte, pero pierde información sobre su localización exacta.

### Average pooling

Promedia activaciones locales y puede representar persistencia más que eventos extremos.

### Global Average Pooling

Promedia cada feature map a lo largo de todo el eje temporal.

Ventajas:

- reduce parámetros;
- evita una capa densa muy grande;
- puede reducir overfitting.

Limitaciones:

- elimina gran parte de la localización temporal;
- trata de forma similar activaciones ocurridas en momentos distintos;
- puede perjudicar targets sensibles al timing.

### Flatten

Conserva separadas las posiciones antes de entrar en una capa densa, pero puede generar muchos parámetros.

Pooling, GAP y Flatten deben compararse experimentalmente.

## 5. Campo receptivo

El campo receptivo de una activación representa cuántos pasos de la entrada pueden influir en ella.

Aumenta mediante:

- kernels mayores;
- más capas;
- pooling;
- stride;
- convoluciones dilatadas.

El campo receptivo local no necesita cubrir obligatoriamente los 30, 60 o 90 minutos completos si las capas posteriores agregan activaciones de varias posiciones.

Debe evaluarse si la arquitectura puede combinar:

- patrones locales;
- contexto distante;
- posición dentro de la ventana;
- información del momento del día.

Las convoluciones dilatadas y las TCN se estudiarán como alternativas posteriores para ampliar el campo receptivo de forma causal.

## 6. Ejemplo CNN1D del libro

El ejemplo financiero univariante utiliza:

- doce retornos mensuales rezagados;
- una entrada de forma `(12, 1)`;
- predicción del retorno del mes siguiente;
- datos de más de 1.500 acciones;
- múltiples divisiones temporales.

La arquitectura incluye:

- `Conv1D` con 32 filtros;
- kernel de tamaño 4;
- activación ReLU;
- padding causal;
- regularización L1/L2;
- max pooling de tamaño 4;
- Flatten;
- Batch Normalization;
- salida lineal.

El entrenamiento utiliza cinco años para pronosticar el mes siguiente y repite el proceso temporalmente.

El libro indica que el modelo suele entrenarse pocas épocas y extrae una cantidad limitada de información sistemática. También advierte que seleccionar las mejores épocas produce sesgo positivo, por lo que los resultados son ilustrativos. 

Este ejemplo no es intradía ni multivariado.

## 7. CNN-TA

CNN-TA convierte factores financieros en una cuadrícula bidimensional.

El enfoque incluye:

- 15 indicadores técnicos y factores;
- cada uno calculado para 15 periodos;
- una matriz de 15 × 15;
- clustering jerárquico para ubicar cerca variables con comportamiento similar;
- una CNN2D que procesa la matriz como una imagen.

La implementación del capítulo utiliza acciones estadounidenses diarias y genera señales para una estrategia long-short transversal. :contentReference[oaicite:3]{index=3}

Limitaciones para MNQ:

- la proximidad espacial es construida artificialmente;
- la cuadrícula puede cambiar entre folds;
- los ejes no poseen necesariamente una interpretación económica natural;
- la transformación debe ajustarse únicamente con el train;
- la estrategia transversal no existe en un instrumento único.

CNN-TA queda como experimento futuro de baja prioridad.

## 8. CNN1D multivariada para MNQ

La adaptación principal tendrá la forma:

```text
muestras × pasos temporales × canales
```

Donde:

* pasos temporales: 30, 60 o 90 minutos;
* canales: OHLCV transformado, indicadores y contexto causal.

La convolución se desplaza únicamente sobre el tiempo.

Los canales se combinan dentro de cada kernel. El orden de las features no constituye un segundo eje espacial.

Una arquitectura inicial controlada podría contener:

1. una o dos capas Conv1D;
2. activación ReLU;
3. pooling opcional;
4. Flatten o Global Average Pooling;
5. una capa densa pequeña;
6. capa de salida específica del target.

La arquitectura debe mantenerse pequeña antes de probar profundidad, conexiones residuales o convoluciones dilatadas.

## 9. Representación de inputs

Debe compararse:

### OHLCV con niveles

Riesgos:

* memorizar el nivel nominal del precio;
* identificar indirectamente el año o contrato;
* aprender discontinuidades de rollover;
* baja estabilidad entre periodos.

### Representaciones relativas

Candidatos:

* retornos;
* diferencias respecto del cierre anterior;
* OHLC relativo al inicio de la ventana;
* rangos normalizados;
* cuerpo y mechas;
* volatilidad;
* ATR;
* volumen relativo;
* posición dentro del rango;
* indicadores técnicos causales.

### Conjuntos experimentales

* OHLCV puro;
* OHLCV transformado;
* secuencia de features completa;
* dataset sin niveles;
* features compactas seleccionadas.

No debe asumirse que incorporar más canales mejora el resultado.

## 10. Normalización

Cada canal puede presentar escala y distribución diferentes.

Los parámetros deben ajustarse exclusivamente con el train:

* media y desviación;
* mínimos y máximos;
* cuantiles;
* límites de outliers;
* normalización del volumen por minuto del día.

Debe definirse si el escalado se realiza:

* globalmente por feature dentro del train;
* por régimen;
* mediante normalización relativa dentro de cada ventana;
* combinando ambos métodos sin utilizar información futura.

Las secuencias no deben cruzar:

* días de trading;
* contratos cuando esté prohibido;
* límites entre train y validación;
* periodos sin datos válidos.

## 11. Salidas y pérdidas

### Regresión

Para retorno, magnitud o volatilidad futura:

* salida lineal;
* MSE como baseline;
* MAE o Huber como alternativas.

### DIR binario

* una unidad;
* sigmoide;
* binary cross-entropy.

### OPC multiclase

* cinco unidades;
* softmax;
* sparse categorical cross-entropy o categorical cross-entropy.

### BAR

La salida dependerá de su formulación concreta:

* binaria;
* multiclase;
* regresión.

Las probabilidades producidas por sigmoide o softmax no están automáticamente calibradas.

## 12. Regularización

Técnicas candidatas:

* early stopping;
* L1/L2 en kernels;
* dropout en capas densas;
* reducción de filtros;
* reducción de capas;
* pooling;
* Batch Normalization.

No deben acumularse automáticamente.

Batch Normalization debe compararse con una arquitectura sin ella porque:

* depende de estadísticas de minibatch;
* puede ser sensible al batch size;
* mantiene estadísticas móviles para inferencia;
* puede verse afectada por cambios de distribución.

Como propuesta moderna puede estudiarse Layer Normalization, pero no es necesaria para el baseline inicial.

## 13. Validación temporal

Para cada fold:

1. construir secuencias causales dentro del train;
2. ajustar preprocesamiento exclusivamente con train;
3. reservar validación interna cronológicamente posterior;
4. ajustar arquitectura y early stopping internamente;
5. congelar el modelo;
6. evaluar una sola vez sobre el periodo externo.

La evaluación externa no debe utilizarse para elegir:

* lookback;
* kernel size;
* filtros;
* pooling;
* padding;
* normalización;
* arquitectura;
* semilla;
* número de épocas.

## 14. Contexto temporal y regímenes

La compartición de filtros puede ocultar que la ubicación horaria importa.

Deben compararse:

* CNN global sin contexto;
* CNN global con `minute_of_day`;
* CNN global con `regime_id`;
* modelos separados por régimen.

Los cinco regímenes actuales son:

* Overnight;
* Pre-market;
* Opening;
* Regular;
* Closing.

Debe analizarse si una mejora por régimen proviene de señal real o simplemente de diferencias en distribución, volumen y volatilidad.

## 15. Métricas

### Rendimiento predictivo

* macro F1;
* balanced accuracy;
* log loss;
* matriz de confusión;
* métricas por clase;
* MAE o RMSE para regresión.

### Calibración

* Brier binario o multiclase;
* curvas de calibración;
* error de calibración;
* confianza por clase.

### Rendimiento financiero

* P&L bruto;
* P&L neto;
* costes;
* turnover;
* Sharpe;
* drawdown;
* estabilidad por año y régimen.

Estas dimensiones deben permanecer separadas.

## 16. Contenido de imágenes no transferible

No son directamente aplicables al MNQ actual:

* transfer learning desde ImageNet;
* VGG16;
* AlexNet;
* clasificación EuroSat;
* detección YOLO;
* segmentación semántica;
* rotación y volteo de imágenes.

Como propuesta futura podría investigarse preentrenamiento autosupervisado sobre series financieras, pero es distinto del transfer learning desde imágenes.

Las aumentaciones temporales deben excluirse del baseline y solo estudiarse si conservan:

* coherencia OHLC;
* orden temporal;
* distribución financiera;
* significado del target;
* reglas de barreras.

## 17. Comparaciones experimentales

La CNN1D debe compararse con:

* MLP;
* regresión logística;
* Random Forest;
* LightGBM;
* XGBoost;
* CatBoost;
* posteriormente LSTM;
* GRU;
* TCN.

Ablaciones recomendadas:

* 30 vs. 60 vs. 90 minutos;
* OHLCV puro vs. transformado;
* uno vs. varios bloques convolucionales;
* kernels pequeños vs. grandes;
* pooling vs. no pooling;
* Flatten vs. GAP;
* Batch Normalization vs. ninguna;
* modelo global vs. modelos por régimen.

## 18. Decisiones pendientes

* Definir canales de entrada.
* Elegir representación con o sin niveles.
* Definir normalización causal.
* Seleccionar arquitectura inicial mínima.
* Elegir kernel y número de filtros.
* Comparar padding.
* Decidir pooling o GAP.
* Establecer protocolo de semillas.
* Medir coste CPU/GPU.
* Diseñar calibración OOF.
* Comparar rendimiento entre regímenes.
* Verificar que ninguna secuencia atraviese límites prohibidos.

## 19. Riesgos metodológicos

* considerar causal padding como protección completa contra leakage;
* introducir información futura durante el escalado;
* usar niveles de precios que identifiquen periodos;
* cruzar sesiones o folds;
* diseñar una red excesivamente profunda;
* asumir que patrones locales existen;
* perder timing mediante pooling;
* memorizar minuto del día o contrato;
* interpretar CNN-TA como una imagen económica natural;
* seleccionar arquitectura sobre evaluación externa;
* asumir probabilidades calibradas;
* confundir resultados mensuales o diarios del libro con MNQ intradía;
* asumir que CNN supera a boosting o modelos recurrentes.