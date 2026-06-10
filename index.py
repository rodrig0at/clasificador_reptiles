import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Clases en el mismo orden que train_generator.class_indices
CLASS_NAMES = [
    "Chameleon",
    "Crocodile_Alligator",
    "Frog",
    "Gecko",
    "Iguana",
    "Lizard",
    "Salamander",
    "Snake",
    "Toad",
    "Turtle_Tortoise"
]

def load_model(model_path):
    print(f"Cargando modelo desde: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0               
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict(modelo, img_array, top_n):
    predictions = modelo.predict(img_array, verbose=0)[0]  
    top_indices = np.argsort(predictions)[::-1][:top_n]

    print("\n── Resultados ──────────────────────")
    for i, idx in enumerate(top_indices):
        label = CLASS_NAMES[idx]
        confidence = predictions[idx] * 100
        marker = "" if i == 0 else " "
        print(f"{marker} #{i+1}  {label:<25} {confidence:.2f}%")
    print("────────────────────────────────────")

def main():
    parser = argparse.ArgumentParser(
        description="Clasificador de reptiles y anfibios"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Ruta a la imagen que se quiere clasificar"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mejor_modelo.keras",
        help="Ruta al modelo guardado (default: mejor_modelo.keras)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Cuántas predicciones mostrar (default: 3)"
    )

    args = parser.parse_args()

    modelo     = load_model(args.model)
    img_array  = preprocess_image(args.image)
    predict(modelo, img_array, args.top)

if __name__ == "__main__":
    main()