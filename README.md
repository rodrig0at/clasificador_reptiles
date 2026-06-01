# 🦎 Clasificador de Reptiles y Anfibios

Clasificador de imágenes entrenado con una CNN personalizada inspirada en la arquitectura optimizada propuesta por Lv et al. (2022). Distingue entre **10 clases** de reptiles y anfibios a partir de fotografías.

---

## Clases

| Índice | Clase |
|--------|-------|
| 0 | Chameleon |
| 1 | Crocodile / Alligator |
| 2 | Frog |
| 3 | Gecko |
| 4 | Iguana |
| 5 | Lizard |
| 6 | Salamander |
| 7 | Snake |
| 8 | Toad |
| 9 | Turtle / Tortoise |

---

## Dataset

Raijin. (2022). *Reptiles and Amphibians Image Dataset*. Kaggle.  
https://www.kaggle.com/datasets/vencerlanz09/reptiles-and-amphibians-image-dataset

- **6,045 imágenes** en total distribuidas en 10 clases
- El dataset está **desbalanceado**: desde 210 imágenes (Chameleon) hasta 1,862 (Turtle_Tortoise)
- El desbalance se maneja con `class_weight` durante el entrenamiento, para penalizar más los errores en clases pequeñas sin descartar ninguna imagen

| Clase | Imágenes |
|-------|----------|
| Turtle_Tortoise | 1,862 |
| Crocodile_Alligator | 692 |
| Lizard | 500 |
| Snake | 500 |
| Frog | 499 |
| Iguana | 499 |
| Toad | 497 |
| Salamander | 484 |
| Gecko | 302 |
| Chameleon | 210 |

---

### División del dataset (70 / 20 / 10)

Se hacen dos llamadas a `train_test_split` con `stratify` para garantizar que la proporción de cada clase se respete en los tres conjuntos.

```python
# Primera división: 70% train / 30% temp
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['clase'], random_state=42)
# Segunda división: 20% val / 10% test
val_df, test_df = train_test_split(temp_df, test_size=0.333, stratify=temp_df['clase'], random_state=42)
```

| Split | Imágenes | Porcentaje |
|-------|----------|------------|
| Train | 4,230 | 70% |
| Validación | 1,209 | 20% |
| Test | 605 | 10% |

### Data Augmentation

El augmentation se aplica **únicamente al conjunto de train**. Val y test solo reciben normalización (`rescale=1./255`).

| Transformación | Valor |
|----------------|-------|
| `zoom_range` | 0.17 |
| `rotation_range` | 7° |
| `width_shift_range` | 0.17 |
| `height_shift_range` | 0.17 |
| `horizontal_flip` | True |

---

## Modelo

La arquitectura sigue el modelo optimizado descrito en Lv et al. (2022), cuya propuesta central consiste en una CNN mejorada sobre VggNet compuesta por **tres capas convolucionales con kernels decrecientes**, seguidas de **tres capas fully connected** y un clasificador **Softmax**. El paper demuestra que esta configuración supera a LeNet, AlexNet y VggNet estándar en accuracy de clasificación de imágenes.

La implementación en este proyecto adapta esa estructura a las características del dataset de reptiles:

```
Input (224 × 224 × 3)
│
├── Conv2D(32, 3×3, relu) → MaxPooling(2×2)
├── Conv2D(64, 3×3, relu) → MaxPooling(2×2)
├── Conv2D(128, 3×3, relu) → MaxPooling(2×2)
│
├── Flatten
├── Dense(256, relu)
├── Dense(128, relu)
├── Dense(64, relu)
│
└── Dense(10, softmax)
```

**Decisiones de diseño basadas en el paper:**

- **3 capas convolucionales + 3 capas FC:** Configuración que el paper identifica como óptima; más capas llevan a overfitting, menos capas degradan la extracción de features.
- **Activación ReLU:** Evita el problema de gradient vanishing que afecta a sigmoides en redes profundas (Lv et al., 2022, Sec. 4.2, Eq. 2).
- **MaxPooling:** El paper recomienda explícitamente *max pooling sampling* para reducir el feature map y evitar overfitting.
- **Softmax en la salida:** Clasificador supervisado que produce probabilidades por clase (Lv et al., 2022, Sec. 3.1, Eq. 1).
- **Data augmentation:** El paper señala que extender el dataset mediante aumentación mejora la capacidad de generalización del modelo (Sec. 4.1).

**Compilación:**

```python
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Resultados

> ⚠️ *Los valores de accuracy y loss se agregarán al finalizar el entrenamiento.*

### Métricas globales

| Métrica | Train | Validación | Test |
|---------|-------|------------|------|
| Accuracy | — | — | — |
| Loss | — | — | — |

### Accuracy por clase (Test Set)

| Clase | Accuracy |
|-------|----------|
| Chameleon | — |
| Crocodile / Alligator | — |
| Frog | — |
| Gecko | — |
| Iguana | — |
| Lizard | — |
| Salamander | — |
| Snake | — |
| Toad | — |
| Turtle / Tortoise | — |

---

## Obstáculos encontrados y soluciones

**El dataset no estaba dividido en train/val/test**  
Las imágenes estaban organizadas únicamente por clase, sin subcarpetas de train, val o test. Se resolvió recolectando las rutas con `os`, construyendo un DataFrame y usando `train_test_split` dos veces. Esto permitió usar `flow_from_dataframe` en lugar de `flow_from_directory`.

**El dataset estaba desbalanceado**  
Al explorar el dataset con `Counter` se descubrió que las clases tenían entre 210 y 1,862 imágenes (diferencia de casi 9×). Reducir todas las clases a 210 imágenes habría limitado demasiado el entrenamiento. Se optó por `class_weight`, que penaliza más los errores en clases pequeñas sin descartar ninguna imagen.

**Las imágenes estaban ordenadas por clase en el DataFrame**  
Sin `stratify`, `train_test_split` habría cortado en medio de una clase. Se solucionó pasando `stratify=df['clase']` en ambas llamadas al split.

---

## Requisitos

```
tensorflow
keras
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Instalación:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## Uso

1. Clona el repositorio y coloca el dataset en la ruta configurada en `data_dir`
2. Abre `main.ipynb` en local o en Google Colab
3. Ejecuta las celdas en orden

---

## Referencias

Lv, Q., Zhang, S., & Wang, Y. (2022). Deep learning model of image classification using machine learning. *Advances in Multimedia*, *2022*, Article 3351256. https://doi.org/10.1155/2022/3351256

Raijin. (2022). *Reptiles and Amphibians Image Dataset*. Kaggle. https://www.kaggle.com/datasets/vencerlanz09/reptiles-and-amphibians-image-dataset
