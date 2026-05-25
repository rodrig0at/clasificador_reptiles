# 🦎 Clasificador de Reptiles

Clasificador de imágenes de reptiles entrenado con... Distingue entre 10 clases de reptiles a partir de fotografías.

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

Raijin. (2022). Reptiles and Amphibians Image Dataset [https://www.kaggle.com/datasets/vencerlanz09/reptiles-and-amphibians-image-dataset]. Kaggle.

- 6,045 imágenes en total distribuidas en 10 clases
- El dataset está **desbalanceado**: desde 210 imágenes (Chameleon) hasta 1,862 (Turtle_Tortoise)
- Se maneja el desbalance con `class_weight` durante el entrenamiento en lugar de undersampling, para aprovechar todos los datos disponibles

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


###  División del dataset (70 / 20 / 10)

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

El augmentation se aplica **únicamente al conjunto de train**. Val y test solo reciben normalización.



---

## Obstáculos encontrados y soluciones

**El dataset no estaba dividido en train/val/test**
Las imágenes estaban organizadas únicamente por clase, sin subcarpetas de train, val o test. Dividirlas manualmente habría sido tedioso y propenso a errores. Se resolvió recolectando las rutas con `os`, construyendo un DataFrame y usando `train_test_split` dos veces para hacer el split en código. Esto permitió usar `flow_from_dataframe` en lugar de `flow_from_directory`.

**El dataset estaba desbalanceado**
Al explorar el dataset con `Counter` se descubrió que las clases tenían entre 210 y 1,862 imágenes, una diferencia de casi 9x. Se pensó por quitar imágenes y que solo hubiera 210 de cada clase, pero esto a la hora del train haría que el modelo no entrenara de forma óptima por tan pocas imágenes. Por eso se buscó una solución y se encontró `class_weight`, que penaliza más los errores en clases pequeñas sin descartar ninguna imagen.

**Las imágenes estaban ordenadas por clase en el DataFrame**
Al construir el DataFrame todas las imágenes de Chameleon quedaron primero, luego todas las de Gecko, y así sucesivamente. Sin `stratify`, `train_test_split` habría cortado en medio de una clase y algunos conjuntos habrían quedado sin representación de todas las clases. Se solucionó pasando `stratify=df['clase']` en ambas llamadas al split.

---

## Requisitos

```
tensorflow
keras
scikit-learn
pandas
numpy
```

Instalación:

```bash
pip install tensorflow scikit-learn pandas numpy
```

---

## Uso

1. Clona el repositorio y coloca el dataset en la ruta configurada en `data_dir`
2. Abre `main.ipynb` en local o en google colab
3. Ejecuta las celdas en orden
