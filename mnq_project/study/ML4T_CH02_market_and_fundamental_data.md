# Capítulo 2 — Market and Fundamental Data: Sources and Techniques

## 1. Conocimiento explícito del libro

- La microestructura estudia cómo las reglas, participantes, órdenes y centros de negociación influyen en la formación de precios, liquidez y costes.
- El capítulo presenta órdenes de mercado, límite y stop, destacando que poseen diferentes condiciones de activación y ejecución.
- Los ticks llegan de forma irregular y contienen ruido de microestructura. El libro los regulariza mediante barras de tiempo, ticks, volumen o valor negociado.
- Las barras de tiempo agregan uniformemente periodos con niveles de actividad muy diferentes.
- Las barras de ticks, volumen y dólar intentan sincronizar las observaciones con la actividad del mercado, pero su construcción exacta requiere operaciones individuales.
- El bid-ask bounce aparece cuando las operaciones alternan entre el bid y el ask.
- Nasdaq TotalView-ITCH permite reconstruir órdenes, ejecuciones y cancelaciones del mercado de acciones Nasdaq; no representa directamente el mercado de futuros CME.
- AlgoSeek proporciona barras de un minuto para acciones con OHLCV, trades, quotes, spreads, VWAP y datos de dirección de operaciones. :contentReference[oaicite:2]{index=2}
- El libro compara formatos de almacenamiento. En su experimento, Parquet obtiene buenos resultados para datos mixtos y HDF5 presenta ventajas en ciertas lecturas numéricas; la elección depende del caso y del benchmark. :contentReference[oaicite:3]{index=3}
- El capítulo también presenta datos fundamentales corporativos obtenidos de formularios SEC y representados mediante XBRL. :contentReference[oaicite:4]{index=4}

## 2. Implicaciones validadas para MNQ

- Mantener Parquet es coherente con el proyecto; no existe una razón demostrada para migrar a HDF5 sin medir rendimiento, tamaño y mantenimiento.
- Debe documentarse la semántica exacta del timestamp:
  - si etiqueta el inicio o el final de la barra;
  - qué intervalo representa;
  - cuándo están disponibles open, high, low, close y volumen;
  - cuál es la primera barra donde puede ejecutarse una decisión.
- Una feature calculada con el cierre y volumen de una vela solo puede utilizarse después de que esa vela haya terminado.
- Los horarios y cambios de actividad justifican investigar regímenes temporales, pero no validan automáticamente los cinco regímenes actuales.
- Las órdenes límite, stop y take profit necesitan reglas conservadoras cuando solo se dispone de OHLCV.
- Los outliers deben investigarse antes de eliminarlos, distinguiendo errores, rollover, eventos reales y episodios extremos de volatilidad.
- El tratamiento de contratos trimestrales y de la continuidad histórica debe validarse para evitar saltos artificiales.

## 3. Información perdida con OHLCV de un minuto

- No se conoce la secuencia exacta de precios dentro de cada minuto.
- Cuando high y low alcanzan dos barreras opuestas, no puede determinarse cuál se alcanzó primero.
- No se dispone de bid, ask, spread, profundidad, operaciones individuales ni dirección de cada trade.
- El posible efecto del bid-ask bounce puede permanecer incorporado en los precios, pero no puede aislarse ni cuantificarse.
- No puede calcularse el VWAP intraminuto real.
- No puede conocerse el precio exacto al que habría sido ejecutada una orden durante la vela.
- No pueden reconstruirse exactamente barras de ticks, volumen o dólar con límites intraminuto.

## 4. Datos adicionales necesarios

Para estudiar microestructura real en MNQ serían necesarios:

- trades tick-by-tick;
- mejor bid y ask del futuro;
- tamaño disponible en cada lado;
- profundidad del libro de órdenes de CME;
- timestamps de alta resolución;
- número y dirección de las operaciones.

Nasdaq ITCH y NBBO son referencias útiles para comprender acciones, pero no son las fuentes correspondientes al futuro MNQ.

## 5. Experimentos pendientes

- Verificar empíricamente la convención temporal del dataset.
- Construir barras aproximadas de volumen acumulando velas completas de un minuto, reconociendo el exceso sobre el umbral y la pérdida intraminuto.
- Comparar barras temporales con agregaciones de 2, 5 u otros minutos antes de adquirir datos tick.
- Analizar volumen relativo por minuto, régimen y sesión.
- Evaluar BOP y otros indicadores basados en la forma de la vela únicamente como proxies OHLCV, no como order flow.
- Diseñar reglas para velas ambiguas:
  - excluirlas;
  - asumir el resultado adverso;
  - etiquetarlas como ambiguas;
  - analizarlas separadamente.
- Diagnosticar saltos de rollover y comparar series sin ajuste, ajustadas y normalizadas.

## 6. Riesgos metodológicos

- Utilizar el close de una barra para decidir y ejecutar al mismo close.
- Suponer ejecución exacta en una barrera cuando el mercado pudo atravesarla con slippage.
- Elegir arbitrariamente el orden TP/SL cuando ambos aparecen en la misma vela.
- Interpretar indicadores OHLCV como medidas reales de presión compradora o vendedora.
- trasladar directamente conclusiones de acciones Nasdaq fragmentadas al mercado centralizado de futuros CME.
- eliminar movimientos extremos legítimos como si fueran errores.
- construir una serie continua sin considerar correctamente el cambio de contrato.