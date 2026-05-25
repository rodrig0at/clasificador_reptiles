# 🦎 Clasificador de Reptiles

Clasificador de imágenes de reptiles entrenado con ... Distingue entre 10 clases de reptiles a partir de fotografías.

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

## Decisiones de diseño


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
2. Abre `main.ipynb` en Jupyter
3. Ejecuta las celdas en orden
