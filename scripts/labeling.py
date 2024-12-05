import os
import cv2
import shutil
import dlib
from imutils import paths

# Configuraciones y rutas
image_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/alter'
label_dir = 'C:/Users/ferik/Desktop/ProyectoU4/data/label'
labeless_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/labeless'  # Carpeta para imágenes sin rostro

# Crear directorios si no existen
os.makedirs(label_dir, exist_ok=True)
os.makedirs(labeless_dir, exist_ok=True)

# Cargar el detector de rostros de Dlib
detector_dlib = dlib.get_frontal_face_detector()

# Cargar el clasificador Haar Cascade para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para etiquetar una imagen
def label_image(image_path, label_path, gender):
    try:
        print(f"[INFO] Etiquetando la imagen {image_path}")
        # Cargar imagen
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar rostros con Dlib
        faces_dlib = detector_dlib(gray)
        faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Si no se detectan rostros con ninguno de los modelos
        if len(faces_dlib) == 0 and len(faces_haar) == 0:
            print(f"[WARNING] No se encontró rostro en la imagen: {image_path}")
            # Mover la imagen a la carpeta "labeless"
            shutil.move(image_path, os.path.join(labeless_dir, os.path.basename(image_path)))
            return False  # No se etiquetó automáticamente

        # Generar las etiquetas con las coordenadas de los rostros y la clase
        with open(label_path, 'w') as label_file:
            # Usar Dlib para etiquetar los rostros
            for face in faces_dlib:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                label_file.write(f"0 {x} {y} {w} {h}\n")  # '0' es la clase de "rostro"
            
            # Usar Haar Cascade para etiquetar los rostros
            for (x, y, w, h) in faces_haar:
                label_file.write(f"0 {x} {y} {w} {h}\n")  # '0' es la clase de "rostro"

            # Añadir clase de género
            label_file.write(f"Clase: {gender}\n")
        
        print(f"[INFO] Etiquetado realizado para {image_path}")
        return True  # Etiquetado automático exitoso

    except Exception as e:
        print(f"[ERROR] Error al etiquetar la imagen {image_path}: {e}")
        return False

# Función para etiquetar las imágenes del conjunto de entrenamiento
def label_train_images():
    try:
        print("[INFO] Iniciando el etiquetado de imágenes...")
        
        # Leer rutas desde train.txt
        with open('C:/Users/ferik/Desktop/ProyectoU4/data/train.txt', 'r') as file:
            image_paths = file.read().strip().split('\n')

        for image_path in image_paths:
            try:
                # Definir ruta de la etiqueta
                filename = os.path.basename(image_path)
                label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

                # Etiquetar la imagen (hombre o mujer)
                gender = 'hombre' if 'hombre' in image_path else 'mujer'

                # Intentar etiquetado automático
                if not label_image(image_path, label_path, gender):
                    print(f"[INFO] Imagen sin rostro detectado: {image_path}")

            except Exception as e:
                print(f"[ERROR] Error al etiquetar la imagen {image_path}: {e}")

    except Exception as e:
        print(f"[ERROR] Error al etiquetar las imágenes de entrenamiento: {e}")


# Ejecutar el etiquetado
if __name__ == "__main__":
    label_train_images()
