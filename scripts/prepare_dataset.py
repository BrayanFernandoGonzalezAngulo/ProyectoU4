import cv2
import os
import random
import shutil
import numpy as np

# Configura los directorios
input_dir = "images"
output_dir = "data/labels"
augmented_images_dir = "data/augmented_images"
train_dir = "data/train"
valid_dir = "data/valid"
test_dir = "data/test"

# Carga el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Función para crear una caja delimitadora alrededor del rostro
def create_bounding_box(image, face):
    """Crea una caja delimitadora alrededor del rostro detectado"""
    x, y, w, h = face
    return (x, y, w, h)

# Función para agregar ruido a la imagen
def add_noise(image):
    """Agrega ruido aleatorio a la imagen"""
    row, col, ch = image.shape
    gauss = np.random.normal(0, 1, (row, col, ch))  # Genera ruido gaussiano
    noisy = np.uint8(np.clip(image + gauss * 25, 0, 255))  # Aplica el ruido
    return noisy

# Función para agregar una tormenta de arena (simulación de distorsión)
def add_sandstorm(image):
    """Agrega una simulación de tormenta de arena en la imagen"""
    sandstorm = np.random.uniform(0, 1, image.shape[:2])
    image[sandstorm > 0.7] = 0  # Cambia píxeles aleatorios a negro
    return image

# Función para hacer la imagen más oscura
def add_darkness(image):
    """Oscurece la imagen"""
    return cv2.convertScaleAbs(image, alpha=0.5, beta=0)  # Reduce la intensidad de los píxeles

# Función para iluminar la imagen
def add_light(image):
    """Ilumina la imagen"""
    return cv2.convertScaleAbs(image, alpha=1.5, beta=50)  # Aumenta la intensidad de los píxeles

# Función para etiquetar una imagen y devolver las coordenadas del rostro
def label_image(image_path, class_name):
    """Etiqueta una imagen y devuelve las coordenadas del rostro si se detecta uno"""
    try:
        print(f"[DEBUG] Leyendo la imagen {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] No se pudo leer la imagen: {image_path}")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"[DEBUG] No se detectaron rostros en {image_path}.")
            return None

        # Suponemos que se etiqueta el primer rostro detectado
        x, y, w, h = faces[0]

        # Guardar la caja delimitadora y la clase
        label_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        with open(label_path, 'w') as f:
            # Escribir la clase y las coordenadas normalizadas del rostro
            f.write(f"{class_name} {x} {y} {w} {h}\n")

        print(f"[DEBUG] Etiquetada la imagen {image_path} con rostro en ({x}, {y}, {w}, {h})")
        return (x, y, w, h)
    except Exception as e:
        print(f"[ERROR] Error al procesar la imagen {image_path}: {e}")
        return None

# Función para dividir las imágenes en conjuntos de entrenamiento, validación y prueba
def divide_images():
    """Divide las imágenes en los conjuntos de entrenamiento, validación y prueba"""
    # Crear directorios de salida si no existen
    for directory in [train_dir, valid_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Dividir las imágenes en las carpetas correspondientes
    for class_name in ["hombre", "mujer"]:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[ERROR] El directorio de clase {class_name} no se encuentra en {input_dir}.")
            continue

        # Lista de imágenes de la clase
        image_list = [img for img in os.listdir(class_dir) if img.endswith(".jpg") or img.endswith(".png")]
        random.shuffle(image_list)

        # Dividir en 70% entrenamiento, 15% validación, 15% prueba
        num_images = len(image_list)
        num_train = int(0.7 * num_images)
        num_valid = int(0.15 * num_images)
        
        for i, image_name in enumerate(image_list):
            image_path = os.path.join(class_dir, image_name)
            if i < num_train:
                shutil.copy(image_path, os.path.join(train_dir, image_name))
            elif i < num_train + num_valid:
                shutil.copy(image_path, os.path.join(valid_dir, image_name))
            else:
                shutil.copy(image_path, os.path.join(test_dir, image_name))

        print(f"[DEBUG] Imágenes de la clase {class_name} divididas en entrenamiento, validación y prueba.")

# Función para aplicar preprocesado (aumento de datos) a la imagen
def augment_and_save(image_path, class_name):
    """Aplica aumentos de datos y guarda las imágenes aumentadas"""
    try:
        print(f"[DEBUG] Realizando aumento de la imagen {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] No se pudo leer la imagen: {image_path}")
            return

        # Realizar aumentos: ruido, tormenta de arena, oscuridad y luz
        noisy_image = add_noise(image)
        sandstorm_image = add_sandstorm(image)
        dark_image = add_darkness(image)
        light_image = add_light(image)

        # Guardar las imágenes aumentadas
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(augmented_images_dir, class_name, "noise_" + base_name), noisy_image)
        cv2.imwrite(os.path.join(augmented_images_dir, class_name, "sandstorm_" + base_name), sandstorm_image)
        cv2.imwrite(os.path.join(augmented_images_dir, class_name, "dark_" + base_name), dark_image)
        cv2.imwrite(os.path.join(augmented_images_dir, class_name, "light_" + base_name), light_image)

        print(f"[DEBUG] Imagen aumentada guardada en {augmented_images_dir}/{class_name}/")
    except Exception as e:
        print(f"[ERROR] Error al aumentar la imagen {image_path}: {e}")

# Función principal para procesar las imágenes
def process_images():
    """Procesa las imágenes, etiqueta y aplica aumentos"""
    for class_name in ["hombre", "mujer"]:
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"[ERROR] El directorio de clase {class_name} no se encuentra en {input_dir}.")
            continue

        print(f"[DEBUG] Procesando imágenes de la clase {class_name}...")
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if image_path.endswith(".jpg") or image_path.endswith(".png"):
                print(f"[DEBUG] Procesando la imagen {image_name}...")

                # Etiquetar la imagen
                label_image(image_path, class_name)

                # Realizar el preprocesado (aumento de datos)
                augment_and_save(image_path, class_name)

        print(f"[DEBUG] División de imágenes en conjuntos de entrenamiento, validación y prueba...")
        divide_images()

if __name__ == "__main__":
    try:
        print("[DEBUG] Iniciando el procesamiento de imágenes...")
        process_images()
        print("[DEBUG] Finalizado el procesamiento de imágenes.")
    except Exception as e:
        print(f"[ERROR] Error en el proceso completo: {e}")
