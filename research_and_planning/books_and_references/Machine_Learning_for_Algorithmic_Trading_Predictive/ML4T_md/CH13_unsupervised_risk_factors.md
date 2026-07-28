# Capítulo 13 — Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning

## 1. Conocimiento explícito del libro

El aprendizaje no supervisado busca descubrir estructuras o aprender representaciones informativas sin utilizar un target para dirigir el proceso.

El capítulo organiza estas técnicas en dos tareas principales:

1. reducción de dimensionalidad;
2. clustering de observaciones o features.

La reducción de dimensionalidad transforma la matriz original en un conjunto menor de variables. El clustering conserva las variables, pero agrupa elementos según una definición de similitud.

El aumento de dimensiones vuelve más disperso el espacio de features, incrementa la cantidad de datos necesaria y puede dificultar:

- la generalización;
- el uso de distancias;
- la visualización;
- el coste computacional;
- la separación entre señal y ruido.

Esto no implica que todos los modelos se deterioren de forma idéntica ni que la reducción dimensional sea siempre beneficiosa.

## 2. PCA

PCA es una transformación lineal que genera componentes:

- formados por combinaciones lineales de las features originales;
- ordenados por varianza explicada;
- ortogonales y, por tanto, no correlacionados dentro de la muestra de ajuste.

PCA maximiza varianza, no:

- capacidad predictiva;
- estabilidad temporal;
- interpretabilidad económica;
- utilidad financiera;
- conservación del orden secuencial.

La señal predictiva puede encontrarse en componentes de baja varianza que serían descartados por un criterio exclusivamente reconstructivo.

### Selección del número de componentes

Pueden utilizarse como heurísticas:

- gráfico de codo;
- varianza explicada acumulada;
- umbrales como 80 %, 90 %, 95 % o 99 %;
- criterio MLE cuando sea apropiado.

La selección definitiva debe realizarse mediante validación interna temporal comparando el rendimiento fuera de muestra.

## 3. Escalado y preparación

Cuando las features poseen unidades y escalas diferentes, PCA sobre la matriz de covarianza puede quedar dominado por las variables de mayor varianza numérica.

Para MNQ deben evaluarse:

- estandarización;
- transformaciones robustas;
- PCA sobre matriz de correlación;
- PCA separado por familias de features.

No deben incorporarse automáticamente a PCA:

- variables categóricas;
- identificadores;
- variables binarias;
- `regime_id`;
- contrato;
- año;
- minuto del día sin una representación justificada.

Todos los parámetros deben estimarse exclusivamente con el train:

- imputación;
- límites de winsorización;
- tratamiento de outliers;
- escaladores;
- PCA;
- número de componentes.

## 4. ICA

ICA también es una transformación lineal.

Busca componentes estadísticamente independientes, una condición más fuerte que la simple ausencia de correlación.

Sus supuestos habituales incluyen:

- mezcla lineal de fuentes latentes;
- fuentes estadísticamente independientes;
- componentes generalmente no gaussianos;
- como máximo una fuente gaussiana.

ICA no identifica automáticamente:

- señal;
- ruido;
- microestructura;
- tendencia;
- volatilidad;
- factores económicos.

El significado de cada componente debe investigarse posteriormente mediante sus cargas, comportamiento temporal y utilidad fuera de muestra.

ICA debe conservarse como alternativa exploratoria a PCA, no como método automático para separar señal y ruido.

## 5. Estabilidad de componentes

Los componentes pueden cambiar entre folds debido a:

- cambios de signo;
- intercambio de orden;
- rotación entre componentes con autovalores similares;
- cambios estructurales en el mercado;
- diferencias de escala entre periodos.

Una inversión de signo no representa necesariamente inestabilidad.

La evaluación debe considerar:

- varianza explicada;
- similitud absoluta de las cargas;
- correlación entre scores;
- ángulos entre subespacios;
- estabilidad de la utilidad predictiva;
- estabilidad por año y régimen.

Los componentes utilizados en cada evaluación deben haber sido ajustados únicamente con el train correspondiente.

## 6. Manifold learning

El capítulo presenta t-SNE y UMAP como técnicas no lineales para representar datos de alta dimensión.

### t-SNE

- prioriza la conservación de vecindades locales;
- depende de la perplejidad, escalado y semilla;
- puede producir agrupamientos visualmente marcados;
- no conserva bien las distancias globales;
- no permite proyectar nuevos datos de forma natural.

### UMAP

- utiliza relaciones locales y topología difusa;
- suele escalar mejor que t-SNE;
- puede preservar más estructura global en algunos casos;
- permite transformar nuevas observaciones después del ajuste;
- sigue siendo sensible a vecinos, distancia, muestra y semilla.

Para MNQ deben utilizarse principalmente para exploración.

Una visualización no demuestra que:

- existan regímenes económicos reales;
- los cinco regímenes horarios actuales sean óptimos;
- los clusters sean estables;
- la separación posea capacidad predictiva.

## 7. Clustering

El capítulo incluye:

- k-means;
- clustering jerárquico;
- DBSCAN;
- HDBSCAN;
- Gaussian Mixture Models.

### K-means

Agrupa observaciones minimizando la distancia respecto de centroides.

Limitaciones:

- exige elegir el número de clusters;
- favorece grupos aproximadamente esféricos;
- es sensible a escala, outliers e inicialización;
- puede confundir estacionalidad horaria con regímenes.

### Clustering jerárquico

Construye una jerarquía de agrupamientos representada mediante un dendrograma.

Deben definirse explícitamente:

- métrica de distancia;
- método de enlace;
- nivel de corte;
- criterio de selección de clusters.

Para clustering de features pueden utilizarse distancias derivadas de:

- correlación Pearson;
- correlación Spearman;
- correlación absoluta;
- información mutua.

Las correlaciones negativas fuertes también pueden representar redundancia.

El enlace Ward requiere geometría euclidiana. No debe combinarse indiscriminadamente con una matriz de distancia basada en correlaciones.

### DBSCAN y HDBSCAN

Agrupan observaciones según densidad y pueden marcar puntos como ruido.

HDBSCAN permite representar clusters con densidades distintas y proporciona información sobre su estabilidad.

Para MNQ son técnicas exploratorias para:

- estados atípicos;
- sesiones inusuales;
- periodos de volatilidad extrema;
- posibles agrupamientos no esféricos.

Sus resultados no representan automáticamente regímenes financieros.

### Gaussian Mixture Models

GMM supone que los datos proceden de una mezcla de distribuciones gaussianas multivariadas.

Permite asignaciones probabilísticas en lugar de etiquetas rígidas.

Puede explorarse para estados de mercado, pero sus supuestos distributivos y su estabilidad temporal deben verificarse.

## 8. Clustering de features para MNQ

El clustering de columnas puede utilizarse para detectar grupos redundantes, por ejemplo:

- retornos con diferentes lags;
- RSI con distintos periodos;
- ATR y rangos;
- volatilidad rolling y EWMA;
- volumen y volumen relativo;
- momentum de distintos horizontes.

Dentro de cada cluster pueden evaluarse:

- conservar todas las variables;
- elegir una representante;
- construir un promedio;
- aplicar PCA local;
- comparar mediante eliminación controlada.

Si la representante se elige según relación con el target, toda la selección pasa a ser supervisada y debe realizarse exclusivamente dentro del train.

## 9. Experimentos aplicables

Deben compararse al menos estas representaciones:

1. features originales;
2. selección manual por conocimiento del dominio;
3. representantes obtenidas mediante clustering;
4. PCA global;
5. PCA por familias;
6. ICA experimental.

La comparación debe realizarse por tipo de modelo:

### Modelos lineales y MLP

Pueden beneficiarse de:

- menor multicolinealidad;
- menor dimensión;
- inputs con escalas más estables.

### Random Forest y boosting

Pueden tolerar redundancia de manera diferente.

PCA puede perjudicarlos al:

- convertir variables dispersas en componentes densos;
- destruir umbrales interpretables;
- mezclar relaciones útiles;
- reducir la capacidad de atribuir importancia.

### Modelos secuenciales

PCA aplicado a una matriz tabular no sustituye el modelado explícito del orden temporal mediante CNN1D, LSTM, GRU o TCN.

## 10. Protocolo temporal

Para cada fold:

1. ajustar imputación y tratamiento de outliers con train;
2. ajustar escalado con train;
3. ajustar PCA, ICA o clustering con train;
4. transformar validación interna sin reajustar;
5. seleccionar configuración dentro del train;
6. congelar la transformación;
7. aplicar a la evaluación externa.

La evaluación externa no debe utilizarse para elegir:

- número de componentes;
- umbral de varianza;
- métrica de distancia;
- cantidad de clusters;
- feature representante;
- hiperparámetros de t-SNE, UMAP o clustering.

## 11. Contenido multiactivo no transferible

El capítulo aplica PCA a retornos de múltiples acciones para obtener:

- factores de riesgo;
- eigenportfolios;
- exposiciones comunes.

También utiliza clustering jerárquico para Hierarchical Risk Parity.

Estas aplicaciones no se trasladan directamente al MNQ porque actualmente existe:

- un único instrumento;
- una única curva principal de precios;
- ausencia de una cartera multiactivo.

No deben tratarse automáticamente como activos independientes:

- ventanas;
- regímenes;
- modelos;
- targets;
- señales del mismo MNQ.

Una posible combinación futura requeriría demostrar diversificación económica, estabilidad y baja dependencia fuera de muestra.

## 12. Métricas de comparación

### Representación

- número de componentes o features;
- varianza explicada;
- tiempo de transformación;
- memoria utilizada;
- estabilidad entre folds.

### Rendimiento predictivo

- macro F1;
- balanced accuracy;
- log loss;
- MAE o RMSE;
- métricas por clase;
- estabilidad por año y régimen.

### Rendimiento financiero

- P&L neto;
- turnover;
- Sharpe;
- drawdown;
- estabilidad entre folds.

Una representación reducida solo aporta valor si mejora o conserva la generalización, simplifica el modelo o reduce recursos sin destruir información útil.

## 13. Decisiones pendientes

- Cuantificar la redundancia real de la matriz actual.
- Determinar si la dimensión justifica aplicar PCA.
- Definir familias de features.
- Elegir métricas de similitud entre columnas.
- Comparar PCA global y PCA por familias.
- Diseñar la evaluación de estabilidad de subespacios.
- Determinar qué modelos podrían beneficiarse.
- Medir reducción de memoria y tiempo.
- Definir el papel exploratorio de UMAP y t-SNE.
- Evaluar si clustering aporta información adicional a los regímenes horarios existentes.

## 14. Riesgos metodológicos

- aplicar reducción dimensional sin necesidad demostrada;
- ajustar transformaciones con toda la serie;
- seleccionar componentes mediante evaluación externa;
- interpretar varianza como señal;
- interpretar componentes como factores económicos sin evidencia;
- tratar cambio de signo como inestabilidad;
- utilizar Ward con una distancia incompatible;
- interpretar una visualización como validación;
- confundir clusters estadísticos con regímenes económicos;
- seleccionar representantes usando el target externo;
- asumir que PCA ayuda por igual a todos los modelos;
- perder señales predictivas de baja varianza;
- sacrificar interpretabilidad sin una mejora fuera de muestra.