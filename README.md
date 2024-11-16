# Proyecto 1: Integración de subword embeddings en redes neuronales recurrentes (RNNs) para procesamiento de lenguaje natural


| **Nombre del Estudiante** | **Código del Estudiante** |
|---------------------------|---------------------------|
| Fiorella Meza             | 20192730G        |

Este proyecto implementa un modelo de Redes Neuronales Recurrentes (RNN) para la clasificación de intenciones, utilizando subword embeddings generados mediante Byte Pair Encoding (BPE). El objetivo es predecir la intención de un texto dado en lenguaje natural.



## Tabla de Contenidos

- [Introducción](#introducción)
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Decisiones Técnicas](#decisiones-técnicas)
  - [Preprocesamiento de Datos](#preprocesamiento-de-datos)
  - [Embeddings de Subpalabras (BPE)](#embeddings-de-subpalabras-bpe)
  - [Arquitectura del Modelo](#arquitectura-del-modelo)
  - [Métricas de Evaluación](#métricas-de-evaluación)
- [Resultados y Análisis](#resultados-y-análisis)
- [Ejecución del Código](#ejecución-del-código)
  - [Requisitos Previos](#requisitos-previos)
  - [Instrucciones](#instrucciones)
- [Posibles Mejoras](#posibles-mejoras)
- [Conclusiones](#conclusiones)

---

## Introducción

La clasificación de intenciones es una tarea fundamental en el procesamiento del lenguaje natural (NLP) que consiste en identificar la intención detrás de una expresión en lenguaje natural. Esto es especialmente útil en aplicaciones como asistentes virtuales, chatbots y sistemas de respuesta automática.

Este proyecto aborda la clasificación de intenciones en español, aprovechando técnicas modernas de NLP como el uso de embeddings de subpalabras y modelos RNN.

## Descripción del Proyecto

El proyecto utiliza el conjunto de datos **MASSIVE** en español, el cual contiene ejemplos de expresiones junto con sus intenciones correspondientes. El flujo general del proyecto es el siguiente:

1. **Carga y preprocesamiento de datos**: Limpieza y preparación del texto para el modelo.
2. **Construcción de vocabulario**: Creación de un vocabulario de subpalabras utilizando BPE.
3. **Definición y entrenamiento del modelo**: Implementación de un modelo RNN (LSTM) para la clasificación.
4. **Evaluación y predicción**: Evaluación del modelo en datos de validación y realización de predicciones de ejemplo.

## Decisiones Técnicas

### Preprocesamiento de Datos

- **Minúsculas y eliminación de puntuación**: Se convierten todos los textos a minúsculas y se eliminan caracteres especiales para reducir la dimensionalidad y la variabilidad del texto.
- **Tokenización**: Se divide el texto en palabras para facilitar su procesamiento.
- **Eliminación de *stopwords***: Se eliminan palabras comunes en español que no aportan significado específico ("el", "la", "y", etc).
- **Pseudo-lemmatización**: Se realiza una simplificación de sufijos comunes para reducir variantes de la misma palabra (ejem: "corriendo" -> "corr").

Estas técnicas reducen el ruido en los datos y mejoran la capacidad del modelo para generalizar.

### Embeddings de Subpalabras (BPE)

- **Byte Pair Encoding (BPE)**: Se utiliza BPE para generar embeddings de subpalabras, lo que permite manejar palabras desconocidas y reducir el tamaño del vocabulario.

Los embeddings de subpalabras capturan mejor las relaciones entre palabras y manejan eficientemente palabras raras o desconocidas, especialmente en idiomas con alta morfología como el español.

### Arquitectura del Modelo

- **Modelo RNN (LSTM bidireccional)**:
  - **Embeddings**: Capa de embeddings para representar las subpalabras.
  - **LSTM bidireccional**: Captura dependencias contextuales en ambas direcciones.
  - **Dropout**: Evita el sobreajuste durante el entrenamiento.
  - **Capa Fully Connected**: Proporciona las predicciones finales de clasificación.

Se usó LSTM debido a que son efectivos para modelar secuencias de texto, y la bidireccionalidad permite capturar contexto tanto previo como futuro.

### Métricas de Evaluación

- **Pérdida (Cross-Entropy Loss)**: Mide la diferencia entre las predicciones del modelo y las etiquetas reales.
- **Exactitud (Accuracy)**: Proporción de predicciones correctas sobre el total.
- **Puntuación F1 (F1-Score)**: Media armónica de la precisión y el recall, útil para datos desbalanceados.

Estas métricas proporcionan una visión integral del rendimiento del modelo, tanto en términos generales como considerando la distribución de clases.

## Resultados y Análisis

Durante el entrenamiento de 50 épocas, se observó lo siguiente:

- **Convergencia de la pérdida**: La pérdida de entrenamiento disminuyó rápidamente en las primeras épocas, indicando que el modelo aprendió patrones significativos.
- **Estancamiento en validación**: A partir de la época 4, la pérdida de validación y las métricas de exactitud y F1 se estabilizaron.

- **Rendimiento final**:
  - **Pérdida de validación**: 1.045
  - **Exactitud en validación**: 76.75%
  - **Puntuación F1 en validación**: 76.69%

**Análisis**:

- El modelo alcanza una exactitud razonable, pero se observa un estancamiento temprano, lo que sugiere limitaciones en la capacidad del modelo o en la representatividad de los datos.
- El uso de BPE y embeddings de subpalabras ayuda a manejar palabras desconocidas, pero puede no ser suficiente para capturar todas las variaciones lingüísticas del español.

**Ejemplo de predicción**:

- **Texto**: "Reproduce mi lista de música favorita"
- **Intención predicha**: `play_music`

El modelo es capaz de identificar correctamente la intención del texto de prueba.

## Ejecución del Código

### Requisitos Previos

- **Python 3.10**
- **Bibliotecas**:
  - `torch` (PyTorch)
  - `datasets`
  - `scikit-learn`
  - `numpy`
  - `re`

### Instrucciones


### Requisitos previos

- Python 3.7 o superior
- pip instalado
- Jupyter Notebook o JupyterLab instalado

### Instalación

1. **Clonar el repositorio**:

   ```bash
   git clone https://github.com/fiorellamr/PC3_CC0C2.git
   ```

2. **Crear un entorno virtual**:

   ```bash
   python -m venv venv
   ```

## Uso del Sistema

Para ejecutar el proyecto:

1. **Activar el entorno virtual**:

   ```bash
   source venv/bin/activate
   ```

2. **Iniciar Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

3. **Abrir el notebook**:

   - Abre el archivo `main.ipynb` y ejecutar las celdas.

## Posibles Mejoras

1. **Incrementar el número de fusiones BPE**:

   - Un mayor número de fusiones puede capturar mejor las combinaciones de subpalabras, mejorando la representación del texto.

2. **Ajustar hiperparámetros**:

   - **Tasa de aprendizaje**: Experimentar con tasas de aprendizaje más pequeñas podría permitir un aprendizaje más fino.
   - **Número de épocas**: Aumentar el número de épocas o implementar early stopping basado en métricas de validación.

3. **Regularización adicional**:

   - **Dropout**: Ajustar la tasa de dropout o añadir dropout en otras capas.

4. **Modelo más complejo**:

   - **Capas adicionales**: Agregar más capas LSTM o capas densas puede aumentar la capacidad del modelo.

5. **Uso de modelos pre-entrenados**:

   - **Transformers**: Utilizar modelos como BERT o RoBERTa pre-entrenados en español.
   - **Embeddings pre-entrenados**: Usar embeddings como FastText o GloVe entrenados en grandes corpus en español.

6. **Balanceo de clases**:

   - **Estratificación**: Asegurar que el conjunto de entrenamiento y validación tengan una distribución similar de clases.

7. **Data Augmentation**:

   - **Sinónimos y paraphraseo**: Generar nuevas muestras variando las existentes para aumentar la diversidad del conjunto de entrenamiento.

## Conclusiones

El modelo implementado demuestra la viabilidad de utilizar RNN con embeddings de subpalabras para la clasificación de intenciones en español. Sin embargo, los resultados indican que da lugar a mejoras significativas.

Este proyecto es base para futuras exploraciones en el procesamiento del lenguaje natural en español y destaca la importancia de adaptar las técnicas a las características específicas del idioma y la tarea.
