# neural_profit
Repositorio de notebooks para proyecto de modelo predictivo

```mermaid
graph TD
    A[Históricos .txt] --> B[Ingesta Raw]
    B --> C[Preparación Intradía]
    C --> D[Validación de Datos]
    D --> E[Feature Engineering]
    E --> F[Train-Ready Dataset]
    F --> G[Entrenamiento Modelos]
    G --> H[Evaluación y Selección]
