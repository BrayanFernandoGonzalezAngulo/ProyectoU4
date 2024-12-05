import os
import random
import shutil
import cv2
import numpy as np
from skimage.util import random_noise

# Directorios
image_dirs = {
    "hombre": "images/hombre",
    "mujer": "images/mujer",
    "augmented_images": "images/augmented_images"
}

# Funciones de aumento de datos
def increase_brightness(image):
    """Aumentar el brillo de la imagen."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * 1.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error en increase_brightness: {e}")
        return image

def decrease_brightness(image):
    """Disminuir el brillo de la imagen."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * 0.5
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error en decrease_brightness: {e}")
        return image

def add_noise(image):
    """Agregar ruido aleatorio a la imagen."""
    try:
        noisy_image = random_noise(image, mode='s&p', amount=0.1) * 255
        return noisy_image.astype(np.uint8)
    except Exception as e:
        print(f"Error en add_noise: {e}")
        return image

def add_sandstorm(image):
    """Aplicar un efecto de tormenta de arena simulada."""
    try:
        noise = np.random.randint(0, 255, (image.shape[0], image.shape[1], 1), dtype=np.uint8)
        noise = cv2.GaussianBlur(noise, (15, 15), 0)
        return cv2.addWeighted(image, 0.7, noise, 0.3, 0)
    except Exception as e:
        print(f"Error en add_sandstorm: {e}")
        return image

# Función para crear aumento de imágenes
def augment_image(image, img_path, augment_type):
    """Generar imágenes aumentadas y guardarlas en la carpeta adecuada."""
    try:
        if augment_type == "brightness_up":
            augmented = increase_brightness(image)
        elif augment_type == "brightness_down":
            augmented = decrease_brightness(image)
        elif augment_type == "noise":
            augmented = add_noise(image)
        elif augment_type == "sandstorm":
            augmented = add_sandstorm(image)
        
        filename = os.path.join(image_dirs['augmented_images'], f"{augment_type}_{os.path.basename(img_path)}")
        cv2.imwrite(filename, augmented)
        print(f"[DEBUG] Imagen aumentada guardada: {filename}")
    except Exception as e:
        print(f"Error en augment_image (tipo: {augment_type}): {e}")

# Función para dividir los datos
def split_data():
    """Dividir las imágenes en entrenamiento, validación y prueba."""
    try:
        all_images = []
        for folder in ["hombre", "mujer"]:
            img_folder = os.path.join(image_dirs[folder])
            img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]
            all_images.extend(img_files)
        
        # Dividir las imágenes en 70% entrenamiento, 15% validación, 15% prueba
        random.shuffle(all_images)
        train_size = int(len(all_images) * 0.7)
        valid_size = int(len(all_images) * 0.15)

        train_images = all_images[:train_size]
        valid_images = all_images[train_size:train_size + valid_size]
        test_images = all_images[train_size + valid_size:]

        # Guardar las rutas de las imágenes en archivos .txt
        with open('data/train.txt', 'w') as f:
            for img in train_images:
                f.write(img + '\n')
        print(f"[DEBUG] Rutas de entrenamiento guardadas en 'data/train.txt'")

        with open('data/valid.txt', 'w') as f:
            for img in valid_images:
                f.write(img + '\n')
        print(f"[DEBUG] Rutas de validación guardadas en 'data/valid.txt'")

        with open('data/test.txt', 'w') as f:
            for img in test_images:
                f.write(img + '\n')
        print(f"[DEBUG] Rutas de prueba guardadas en 'data/test.txt'")

    except Exception as e:
        print(f"Error en split_data: {e}")

# Función principal
def main():
    """Función principal para ejecutar el preprocesamiento de datos."""
    try:
        if not os.path.exists(image_dirs['augmented_images']):
            os.makedirs(image_dirs['augmented_images'])
            print(f"[DEBUG] Carpeta de imágenes aumentadas creada en: {image_dirs['augmented_images']}")

        # Realizar el aumento de datos en las imágenes existentes
        for folder in ["hombre", "mujer"]:
            img_folder = os.path.join(image_dirs[folder])
            for img_name in os.listdir(img_folder):
                img_path = os.path.join(img_folder, img_name)
                if img_name.endswith(('.jpg', '.png')):
                    print(f"[DEBUG] Procesando imagen: {img_name}")
                    image = cv2.imread(img_path)
                    augment_image(image, img_path, "brightness_up")
                    augment_image(image, img_path, "brightness_down")
                    augment_image(image, img_path, "noise")
                    augment_image(image, img_path, "sandstorm")

        # Dividir los datos en entrenamiento, validación y prueba
        split_data()

    except Exception as e:
        print(f"Error en main: {e}")

if __name__ == "__main__":
    main()
