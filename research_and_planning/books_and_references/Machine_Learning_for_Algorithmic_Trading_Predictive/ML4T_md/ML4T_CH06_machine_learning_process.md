# Capítulo 6 — The Machine Learning Process

## 1. Conocimiento explícito del libro

- El aprendizaje automático busca aprender una relación entre features y un target que generalice a observaciones no vistas.
- El error de generalización depende del equilibrio entre:
  - bias: incapacidad del modelo para representar correctamente la relación;
  - variance: sensibilidad excesiva a las particularidades y al ruido del entrenamiento.
- Los datos financieros suelen violar el supuesto IID debido a dependencia temporal, heterocedasticidad y cambios en su distribución.
- La validación aleatoria convencional puede producir look-ahead cuando altera el orden temporal o mezcla observaciones relacionadas.
- El libro presenta validación temporal expansiva o móvil mediante `TimeSeriesSplit` y divisores personalizados.
- Purging elimina muestras de entrenamiento cuyos intervalos de evaluación se solapan con validación.
- Embargoing elimina muestras de entrenamiento posteriores a un periodo de prueba cuando pueden producir contaminación.
- La validación combinatoria genera más trayectorias históricas que un único walk-forward, aplicando purging y embargoing cuando corresponda.
- Un hold-out final debe permanecer fuera de la selección de modelos e hiperparámetros.
- Las transformaciones aprendidas de los datos deben ajustarse dentro de cada split de entrenamiento, mediante pipelines y validación cruzada adecuada.

## 2. Diseño de validación aplicable a MNQ

La arquitectura experimental debe distinguir:

1. **Train:** ajuste de parámetros del modelo.
2. **Validación interna:** tuning, comparación de configuraciones y early stopping.
3. **Evaluación walk-forward externa:** medición fuera de muestra de cada fold.
4. **Test final:** periodo intacto utilizado una sola vez después de cerrar el protocolo.

Los años de evaluación externa dejan de ser completamente independientes si sus resultados se utilizan reiteradamente para cambiar features, targets, modelos o hiperparámetros.

Purging debe decidirse comprobando los intervalos de cada etiqueta:

- tiempo de predicción;
- comienzo del horizonte futuro;
- final del horizonte utilizado para conocer el target;
- comienzo de la validación.

Si el corte se realiza entre sesiones y los targets no cruzan días, puede no existir solapamiento. Si el corte ocurre dentro de una sesión, deberán retirarse las muestras cuyo horizonte alcance la validación.

Embargoing tiene baja aplicabilidad en un walk-forward donde todo el entrenamiento ocurre antes de la evaluación. No debe decidirse mediante autocorrelación de etiquetas o residuos.

## 3. Preprocesamiento y prevención de leakage

Dentro de cada train deben ajustarse:

- escaladores y normalizadores;
- imputadores;
- PCA u otras reducciones;
- selección de features;
- cuantiles o transformaciones basadas en distribuciones históricas;
- técnicas de balanceo o resampling;
- pesos de clase cuando dependan de la distribución del fold.

`Pipeline` evita que estas transformaciones utilicen la validación, pero no reemplaza un divisor temporal correcto.

Las secuencias de 30, 60 y 90 minutos deben construirse sin atravesar límites prohibidos de sesiones o splits.

## 4. Baselines y métricas

Los baselines deben adaptarse al target:

### Clasificación

- clase mayoritaria o predicción constante;
- probabilidades basadas en frecuencias del train;
- regresión logística;
- reglas simples de momentum o reversión.

Métricas candidatas:

- macro F1 y métricas por clase;
- balanced accuracy;
- log loss;
- matriz de confusión;
- ROC-AUC multiclase con estrategia y promedio explícitos;
- PR-AUC por clase y agregaciones macro o ponderadas.

Ninguna métrica es universal. Los promedios ponderados pueden ocultar un desempeño deficiente en clases minoritarias.

### Regresión

- predicción constante mediante media o mediana del train;
- modelo lineal regularizado;
- MAE;
- RMSE;
- error y correlación de la señal fuera de muestra;
- IC cuando la predicción y el resultado sean continuos u ordenables.

El rendimiento predictivo debe mantenerse separado de la calibración y del rendimiento financiero posterior.

## 5. Decisiones pendientes

- Determinar la duración adecuada del historial de entrenamiento.
- Comparar entrenamiento expansivo frente a ventanas móviles.
- Definir la cantidad y duración de folds externos.
- Reservar formalmente un periodo final intacto.
- Diseñar la validación interna para cada arquitectura.
- Verificar el solapamiento real de DIR, BAR y OPC en cada corte.
- Evaluar separadamente las ventanas de entrada de 30, 60 y 90 minutos.
- Definir qué métricas corresponden a cada target antes de entrenar.
- Registrar configuraciones, semillas, features, predicciones y resultados para controlar múltiples pruebas.

## 6. Propuestas experimentales

- Evaluar calibración de probabilidades cuando estas se utilicen para filtrar entradas o dimensionar posiciones.
- Utilizar Brier Score binario o su extensión multiclase, junto con curvas de calibración.
- Mantener la validación combinatoria como análisis de robustez futuro, no como requisito inicial.
- Comparar la estabilidad de modelos y features entre años, folds y regímenes.
- Analizar la sensibilidad de los resultados al tamaño del train y a la ubicación de los cortes temporales.

## 7. Riesgos principales

- utilizar la evaluación externa para early stopping o tuning;
- consultar repetidamente el test final;
- ajustar preprocesamiento antes de dividir temporalmente;
- balancear clases usando información de validación;
- ignorar horizontes solapados;
- tratar observaciones de un minuto como independientes;
- seleccionar modelos después de probar muchas configuraciones sobre los mismos años;
- confundir una mejora predictiva con rentabilidad después de costes;
- elegir una única métrica sin considerar el target y los costes de los distintos errores.