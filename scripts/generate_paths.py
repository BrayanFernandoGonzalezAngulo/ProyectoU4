import os
import random
from imutils import paths

# Configuraciones y rutas
dataset_dir = 'C:/Users/ferik/Desktop/ProyectoU4/images/dataset'
train_txt = 'C:/Users/ferik/Desktop/ProyectoU4/data/train.txt'
test_txt = 'C:/Users/ferik/Desktop/ProyectoU4/data/test.txt'

# Función para dividir las imágenes en train y test
def generate_paths():
    try:
        print("[INFO] Iniciando la generación de paths...")
        image_paths = list(paths.list_images(dataset_dir))
        random.shuffle(image_paths)  # Mezclar imágenes

        # Dividir en 80% train y 20% test
        train_paths = image_paths[:int(len(image_paths) * 0.8)]
        test_paths = image_paths[int(len(image_paths) * 0.8):]

        # Guardar las rutas de las imágenes en archivos .txt
        with open(train_txt, 'w') as f:
            for path in train_paths:
                f.write(path + '\n')

        with open(test_txt, 'w') as f:
            for path in test_paths:
                f.write(path + '\n')

        print(f"[INFO] Se generaron los archivos {train_txt} y {test_txt}")

    except Exception as e:
        print(f"[ERROR] Error en la generación de paths: {e}")

# Ejecutar la generación de paths
if __name__ == "__main__":
    generate_paths()
