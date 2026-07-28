# Capítulo 1 — Financial Machine Learning as a Distinct Subject

## Objetivo del capítulo

El capítulo establece que el **Machine Learning financiero (MLF)** debe tratarse como una disciplina específica. Los mercados presentan baja relación señal-ruido, dependencia temporal, cambios de régimen, competencia entre participantes y un riesgo elevado de falsos descubrimientos. Frente a estas dificultades, López de Prado propone organizar la investigación como una **cadena de producción especializada**, en lugar de depender de investigadores aislados que intenten construir estrategias completas por sí solos.

## Conocimiento explícito del autor

- **ML financiero como disciplina propia:** las herramientas estándar de ML no deben trasladarse mecánicamente a finanzas, porque los datos y los problemas financieros presentan características particulares.
- **Causa central del fracaso:** el **Paradigma de Sísifo**, donde investigadores aislados buscan estrategias completas y terminan frecuentemente en falsos positivos o soluciones convencionales con poca ventaja.
- **Metaestrategia:** los profesionales deben desarrollar procesos capaces de producir y evaluar descubrimientos de forma metódica y reproducible.
- **Investigación mediante backtesting:** el libro identifica esta práctica como un error metodológico; su tratamiento detallado aparece posteriormente, especialmente en los capítulos 8 y 11.
- **Sobreajuste:** la flexibilidad del ML facilita confundir patrones reales con fluctuaciones aleatorias. El autor considera que el sobreajuste es poco ético y que, cuando se realiza conscientemente, constituye fraude científico.
- **Complejidad del algoritmo:** una estrategia exitosa no depende únicamente de utilizar un modelo sofisticado, sino del conjunto completo formado por datos, features, validación, backtesting, ejecución y gestión del riesgo.

## Paradigma de Sísifo y paradigma de metaestrategia

### Paradigma de Sísifo

Describe una organización donde cada investigador trabaja en un silo y debe producir una estrategia completa. Este enfoque obliga a una misma persona a resolver simultáneamente problemas de datos, features, modelos, validación, ejecución y backtesting. La presión por encontrar resultados puede favorecer la selección oportunista y los falsos descubrimientos.

### Paradigma de metaestrategia

Propone organizar la investigación como una fábrica, con funciones especializadas y procedimientos compartidos. El objetivo no es encontrar una fórmula aislada, sino construir un proceso reproducible que genere, evalúe y descarte hipótesis de forma sistemática.

En un proyecto individual, este paradigma puede implementarse mediante una **separación lógica estricta** entre etapas, configuraciones y artefactos, aunque una misma persona desempeñe varios roles.

## Cadena de producción

El autor describe seis estaciones principales:

1. **Curadores de datos:** recopilan, limpian, indexan, almacenan y ajustan los datos. También gestionan particularidades del instrumento, como los cambios de contrato en futuros.
2. **Analistas de features:** transforman datos crudos en señales informativas y construyen una biblioteca de hallazgos. Su función no es desarrollar por sí solos una estrategia completa.
3. **Estrategas:** combinan hallazgos y formulan una teoría que explique el mecanismo económico de la posible ventaja. La estrategia actúa como experimento para probar esa teoría.
4. **Backtesters:** evalúan la estrategia bajo distintos escenarios y consideran la información sobre cómo fue descubierta, incluido el número de pruebas realizadas.
5. **Equipo de despliegue:** integra el prototipo en producción y garantiza que la implementación sea coherente con el diseño validado.
6. **Supervisión de cartera:** gestiona el ciclo de vida de la estrategia: embargo de despliegue, paper trading, graduación, reasignación y retirada.

## Aplicación al proyecto MNQ

El proyecto MNQ puede adoptar la metaestrategia mediante la separación reproducible de:

```text
datos
→ targets
→ features
→ secuencias
→ entrenamiento
→ selección
→ evaluación
→ backtesting
→ despliegue
```

Los cinco regímenes horarios deben tratarse como **hipótesis de segmentación temporal**, no como regímenes económicos demostrados. Su utilidad debe justificarse y validarse empíricamente.

La comparación entre MLP, CNN1D, LSTM, GRU y TCN es metodológicamente válida si:

- se utiliza el mismo protocolo experimental;
- las configuraciones relevantes se definen antes de observar los resultados;
- se registran todos los experimentos;
- no se reutiliza indefinidamente la validación para modificar decisiones;
- existe una evaluación final no utilizada durante el diseño.

## Prácticas actuales que se mantienen

- **Validación walk-forward:** se mantiene como diseño actual, pendiente de auditoría específica con los capítulos 7 y 12.
- **Arquitecturas previstas:** se mantienen MLP, CNN1D, LSTM, GRU y TCN. El capítulo no justifica sustituirlas.
- **Separación por stages y notebooks:** es compatible con la metaestrategia siempre que compartan configuraciones, datasets, protocolos y artefactos reproducibles.
- **Comparación de datasets y modelos:** puede mantenerse si los criterios de evaluación se fijan previamente y todos los resultados quedan registrados.

## Riesgos metodológicos

- **Selección oportunista:** modificar modelos, hiperparámetros o datasets repetidamente después de observar la validación.
- **Ocultamiento de resultados negativos:** conservar solo los experimentos favorables impide conocer el verdadero proceso de selección.
- **Storytelling ex post:** justificar después de los resultados por qué un régimen, una feature o un modelo “debía funcionar”.
- **Acoplamiento entre etapas:** permitir que una notebook de modelo modifique datos, targets o reglas de validación.
- **Confusión entre embargos:** el embargo del ciclo de despliegue no es el mismo que el embargo temporal utilizado para prevenir leakage en validación.

## Decisiones adoptadas

| Concepto | Aplicación al MNQ | Decisión | Prioridad |
|---|---|---:|---:|
| Metaestrategia | Separar lógicamente datos, targets, features, entrenamiento, selección y evaluación. | **INCORPORAR** | ALTA |
| Regímenes intradía | Tratarlos como hipótesis de segmentación que deben validarse empíricamente. | **AUDITAR** | ALTA |
| Registro de experimentos | Guardar configuraciones, métricas, predicciones, artefactos y resultados negativos. | **INCORPORAR** | ALTA |
| Protocolo de selección | Definir criterios antes de comparar los modelos finales. | **INCORPORAR** | ALTA |
| Validación walk-forward | Mantenerla como diseño actual hasta su auditoría en capítulos posteriores. | **MANTENER** | MEDIA |
| Evaluación final ciega | Definir un periodo o flujo de datos no utilizado durante el diseño. | **APLICAR MÁS ADELANTE** | MEDIA |
| Teoría de la estrategia | Documentar el mecanismo económico hipotético que podría generar ventaja. | **INCORPORAR** | MEDIA |

## Documentación y artefactos requeridos

- **Registro reproducible de experimentos**
  - identificador del experimento;
  - fecha y versión del código;
  - dataset y target;
  - fold;
  - modelo e hiperparámetros;
  - semilla;
  - métricas;
  - predicciones;
  - artefactos;
  - resultado favorable o desfavorable.

- **Protocolo de selección**
  - métrica principal;
  - métricas secundarias;
  - reglas de desempate;
  - criterios de exclusión;
  - límite o presupuesto de experimentos;
  - condiciones para congelar una decisión.

- **Memoria de hipótesis**
  - fundamento de los regímenes horarios;
  - mecanismo económico esperado;
  - relación entre features, target y horizonte;
  - supuestos que deben comprobarse.

- **Configuración central**
  - reglas compartidas por todas las notebooks;
  - versiones de datasets;
  - folds;
  - semillas;
  - métricas;
  - rutas de salida.

## Tareas pendientes

1. Definir el formato central del registro de experimentos de Stage 07.
2. Establecer criterios de selección antes de comparar los cinco modelos.
3. Documentar los cinco regímenes como hipótesis verificables.
4. Verificar que las notebooks de modelos no modifiquen datasets, targets o folds.
5. Definir posteriormente una evaluación final ciega sin fijar todavía su extensión.
6. Auditar la continuidad de contratos y el roll al estudiar el Capítulo 2.

## Conceptos que se desarrollarán en capítulos posteriores

- **Estructura de datos, muestreo y roll de futuros:** Capítulo 2.
- **Labeling, triple barrera y meta-labeling:** Capítulo 3.
- **Pesos por concurrencia y uniqueness:** Capítulo 4.
- **Purging y embargo de validación:** Capítulo 7.
- **Feature importance como proceso de investigación:** Capítulo 8.
- **Tuning con validación financiera:** Capítulo 9.
- **Peligros del backtesting:** Capítulo 11.
- **Walk-forward y CPCV:** Capítulo 12.

## Conclusión

El principal aporte del Capítulo 1 al proyecto MNQ es metodológico. No exige modificar todavía targets, features ni arquitecturas. Exige convertir el proyecto en un proceso reproducible, trazable y modular, donde las hipótesis, configuraciones, experimentos y decisiones queden registradas antes de utilizar el backtesting como evaluación final.

La prioridad inmediata es fortalecer la gobernanza experimental de Stage 07: configuración central, registro completo de intentos, separación de etapas y criterios de selección definidos previamente.
