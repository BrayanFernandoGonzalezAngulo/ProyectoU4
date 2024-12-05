import os
import cv2
import numpy as np
from skimage.util import random_noise
from skimage import exposure
import shutil
from imutils import paths
import random

# Configuraciones y rutas
dataset_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/dataset'
alter_image_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/alter'

# Crear directorios si no existen
os.makedirs(alter_image_dir, exist_ok=True)

# Función de aumento: Tormenta de arena
def add_sandstorm(image):
    try:
        noise_img = random_noise(image, mode='s&p', amount=0.1)
        return (255 * noise_img).astype(np.uint8)
    except Exception as e:
        print(f"[ERROR] Error al aplicar tormenta de arena: {e}")
        return image

# Función de aumento: Ruido
def add_noise(image):
    try:
        noise_img = random_noise(image, var=0.01)
        return (255 * noise_img).astype(np.uint8)
    except Exception as e:
        print(f"[ERROR] Error al aplicar ruido: {e}")
        return image

# Función de aumento: Oscuridad
def add_darkness(image):
    try:
        return cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    except Exception as e:
        print(f"[ERROR] Error al aplicar oscuridad: {e}")
        return image

# Función de aumento: Luz
def add_light(image):
    try:
        return cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    except Exception as e:
        print(f"[ERROR] Error al aplicar luz: {e}")
        return image

# Función para procesar todas las imágenes
def preprocess_images():
    try:
        print("[INFO] Iniciando el preprocesamiento de imágenes...")
        image_paths = list(paths.list_images(dataset_dir))
        random.shuffle(image_paths)  # Mezclar imágenes

        for image_path in image_paths:
            try:
                print(f"[INFO] Procesando la imagen {image_path}")
                # Leer la imagen
                image = cv2.imread(image_path)

                # Aumento de datos
                image_sandstorm = add_sandstorm(image)
                image_noise = add_noise(image)
                image_darkness = add_darkness(image)
                image_light = add_light(image)

                # Guardar las imágenes aumentadas
                filename = os.path.basename(image_path)
                cv2.imwrite(os.path.join(alter_image_dir, filename.replace('.jpg', '_sandstorm.jpg')), image_sandstorm)
                cv2.imwrite(os.path.join(alter_image_dir, filename.replace('.jpg', '_noise.jpg')), image_noise)
                cv2.imwrite(os.path.join(alter_image_dir, filename.replace('.jpg', '_darkness.jpg')), image_darkness)
                cv2.imwrite(os.path.join(alter_image_dir, filename.replace('.jpg', '_light.jpg')), image_light)

                print(f"[INFO] Imágenes aumentadas guardadas para {image_path}")

            except Exception as e:
                print(f"[ERROR] Error al procesar la imagen {image_path}: {e}")

    except Exception as e:
        print(f"[ERROR] Error en el preprocesamiento de imágenes: {e}")

# Ejecutar el preprocesado
if __name__ == "__main__":
    preprocess_images()
