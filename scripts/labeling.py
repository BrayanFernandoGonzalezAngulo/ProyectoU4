import os
import cv2
import dlib
from imutils import paths
import shutil

# Configuraciones y rutas
image_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/alter'
label_dir = 'C:/Users/ferik/Desktop/ProyectoU4/data/label'
tagless_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/labeless'  # Carpeta para imágenes sin rostro
os.makedirs(label_dir, exist_ok=True)
os.makedirs(tagless_dir, exist_ok=True)

# Cargar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Función para etiquetar una imagen (detección automática)
def label_image(image_path, label_path, gender):
    try:
        print(f"[INFO] Etiquetando la imagen {image_path}")
        # Cargar imagen
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        faces = detector(gray)
        if len(faces) == 0:
            # Si no hay rostros, mover la imagen a la carpeta de "tagless"
            print(f"[WARNING] No se encontró rostro en la imagen: {image_path}. Moviendo a 'tagless'.")
            shutil.move(image_path, os.path.join(tagless_dir, os.path.basename(image_path)))
            return

        # Generar las etiquetas con las coordenadas de los rostros y la clase
        with open(label_path, 'w') as label_file:
            for face in faces:
                # Obtener coordenadas del rostro
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                # Escribir las coordenadas del rostro y la clase (hombre/mujer)
                label_file.write(f"0 {x} {y} {w} {h}\n")  # '0' es la clase de "rostro"
            # Añadir clase de género
            label_file.write(f"Clase: {gender}\n")
        print(f"[INFO] Etiquetado realizado para {image_path}")

    except Exception as e:
        print(f"[ERROR] Error al etiquetar la imagen {image_path}: {e}")

# Función para etiquetar una imagen manualmente usando el mouse
def manual_label_image(image_path, label_path, gender):
    try:
        print(f"[INFO] Etiquetado manual de la imagen {image_path}")
        # Cargar la imagen
        image = cv2.imread(image_path)
        clone = image.copy()

        # Variables para guardar las coordenadas del rostro
        rect = []

        # Función de callback para el mouse (para seleccionar la región)
        def click_and_crop(event, x, y, flags, param):
            nonlocal rect
            if event == cv2.EVENT_LBUTTONDOWN:
                rect = [(x, y)]  # Primer punto (esquina superior izquierda)
            elif event == cv2.EVENT_LBUTTONUP:
                rect.append((x, y))  # Segundo punto (esquina inferior derecha)
                cv2.rectangle(image, rect[0], rect[1], (0, 255, 0), 2)
                cv2.imshow("image", image)
        
        # Mostrar la imagen y permitir al usuario seleccionar la región
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Verificar que se haya seleccionado una región
        if len(rect) == 2:
            x1, y1 = rect[0]
            x2, y2 = rect[1]
            w, h = abs(x2 - x1), abs(y2 - y1)

            # Guardar las coordenadas en el archivo de etiquetas
            with open(label_path, 'w') as label_file:
                label_file.write(f"0 {x1} {y1} {w} {h}\n")  # '0' es la clase de "rostro"
                label_file.write(f"Clase: {gender}\n")
            print(f"[INFO] Etiquetado manual realizado para {image_path}")
        else:
            print(f"[ERROR] No se seleccionó una región válida para {image_path}")

    except Exception as e:
        print(f"[ERROR] Error al etiquetar manualmente la imagen {image_path}: {e}")

# Función para etiquetar las imágenes del conjunto de entrenamiento
def label_train_images():
    try:
        print("[INFO] Iniciando el etiquetado de imágenes...")
        image_paths = list(paths.list_images(image_dir))

        for image_path in image_paths:
            try:
                # Definir ruta de la etiqueta
                filename = os.path.basename(image_path)
                label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

                # Verificar si la imagen ya tiene una etiqueta
                if os.path.exists(label_path):
                    print(f"[INFO] Ya existe etiqueta para {image_path}")
                    continue  # Saltar si ya tiene etiqueta

                # Etiquetar la imagen (hombre o mujer)
                gender = 'hombre' if 'hombre' in image_path else 'mujer'
                
                # Etiquetado automático o manual si no se detecta rostro
                if os.path.exists(image_path.replace('.jpg', '_sandstorm.jpg')):  # Revisar las imágenes aumentadas
                    label_image(image_path, label_path, gender)
                else:
                    manual_label_image(image_path, label_path, gender)

            except Exception as e:
                print(f"[ERROR] Error al etiquetar la imagen {image_path}: {e}")

    except Exception as e:
        print(f"[ERROR] Error al etiquetar las imágenes de entrenamiento: {e}")

# Ejecutar el etiquetado
if __name__ == "__main__":
    label_train_images()
