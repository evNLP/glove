# Glove: **Clasificación de Texto con LSTM**

## **Introducción**

Este repositorio contiene un proyecto para la clasificación de texto utilizando modelos LSTM y técnicas avanzadas de representación vectorial como los *embeddings* preentrenados GloVe. El objetivo es demostrar cómo las representaciones de palabras afectan la precisión del modelo y explorar la interacción entre modelos recurrentes y vectores semánticos.

El repositorio abarca desde la implementación básica de un modelo LSTM hasta la integración de *embeddings* preentrenados y el análisis comparativo del rendimiento basado en la dimensión de estos vectores.

## **Características del Proyecto**

### **1. Modelo LSTM**
El modelo LSTM implementado incluye:
- **Capa de Embedding**: Puede usar *embeddings* generados desde cero o vectores preentrenados como GloVe.
- **Capa LSTM**: Procesa las representaciones de palabras y captura dependencias secuenciales en el texto.
- **Capa Fully Connected (FC)**: Produce las predicciones finales para la clasificación.

### **2. Entrenamiento**
- **Optimización**: Se utiliza `SGD` con gradiente recortado para garantizar estabilidad.
- **Planificador de tasa de aprendizaje**: `StepLR` para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.
- **Evaluación**: Precisión medida en cada época tanto en los datos de entrenamiento como en los de validación.

### **3. Uso de *Embeddings* Preentrenados**
El modelo es capaz de incorporar vectores GloVe de diferentes dimensiones (50, 100, 300) y ajustarlos al vocabulario del conjunto de datos. Esto permite evaluar cómo los vectores preentrenados impactan la precisión y la eficiencia del modelo.

## **Requisitos**

### Dependencias:
- Python >= 3.8
- PyTorch >= 1.10
- torchtext >= 0.11
- numpy
- Matplotlib

Instalación:
```bash
pip install torch torchtext matplotlib numpy
```

## **Estructura del Repositorio**

```
glove
├── imgs
│   ├── Lote1_b.svg
│   ├── Lote2_b.svg
│   └── Lote3_b.svg
└── nb03.ipynb
```

## **Resultados y Análisis**

### Rendimiento Comparativo:
| **Embeddings** | **Precisión Inicial** | **Precisión Final** | **Mejor Precisión Validación** | **Tiempo por Época** |
|----------------|------------------------|----------------------|---------------------------------|-----------------------|
| Sin GloVe (25) | 0.441                 | 0.879               | 0.879                          | 34-49 segundos        |
| GloVe 50       | 0.684                 | 0.904               | 0.904                          | 31-46 segundos        |
| GloVe 100      | 0.874                 | 0.909               | 0.909                          | 44-80 segundos        |
| GloVe 300      | 0.888                 | 0.914               | 0.918                          | 52-72 segundos        |

### **Conclusiones:**
- Los *embeddings* preentrenados mejoran significativamente la precisión inicial y final del modelo.
- A mayor dimensión de GloVe, mayor precisión, pero con un incremento en el costo computacional.
- GloVe 300 logra el mejor rendimiento, aunque las mejoras son marginales respecto a GloVe 100.

## **Futuras Mejoras**
1. **Incorporación de modelos más avanzados**: como GRU o Transformers.
2. **Fine-tuning** de *embeddings* preentrenados en lugar de mantenerlos congelados.
3. **Evaluación con diferentes conjuntos de datos**: para validar la generalización del enfoque.
