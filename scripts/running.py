import os
import cv2

# Configuración de rutas
darknet_path = r"C:\Users\ferik\Desktop\darknet-master"
cfg_path = r"C:\Users\ferik\Desktop\ProyectoU4\cfg\yolov4-obj.cfg"
data_path = r"C:\Users\ferik\Desktop\ProyectoU4\data\obj.data"
weights_path = r"C:\Users\ferik\Desktop\ProyectoU4\backup\yolov4-obj_best.weights"

def run_darknet(command):
    """
    Función para ejecutar comandos de Darknet.
    """
    try:
        os.chdir(darknet_path)
        os.system(command)
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema al ejecutar Darknet: {e}")

def test_images():
    """
    Función para probar el modelo con imágenes del conjunto de prueba.
    """
    test_images_dir = r"C:\Users\ferik\Desktop\ProyectoU4\images\test"
    output_dir = r"C:\Users\ferik\Desktop\ProyectoU4\results"
    os.makedirs(output_dir, exist_ok=True)

    # Ejecutar detecciones para todas las imágenes de prueba
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        command = f"darknet detector test {data_path} {cfg_path} {weights_path} -dont_show -ext_output {image_path} -out {output_dir}\\result_{image_name}.txt"
        print(f"[INFO] Procesando: {image_name}")
        run_darknet(command)
    print(f"[INFO] Resultados guardados en: {output_dir}")

def use_camera():
    """
    Función para usar la cámara en tiempo real.
    """
    command = f"darknet detector demo {data_path} {cfg_path} {weights_path} -dont_show"
    print("[INFO] Iniciando cámara en tiempo real...")
    run_darknet(command)

def process_image():
    """
    Función para procesar una imagen seleccionada desde el dispositivo.
    """
    image_path = input("Ingrese la ruta completa de la imagen: ").strip()
    if not os.path.isfile(image_path):
        print("[ERROR] La ruta ingresada no es válida. Inténtalo de nuevo.")
        return

    command = f"darknet detector test {data_path} {cfg_path} {weights_path} -dont_show -ext_output {image_path}"
    print("[INFO] Procesando la imagen...")
    run_darknet(command)

def main_menu():
    """
    Menú principal.
    """
    while True:
        print("\n=== Menú Principal ===")
        print("1. Probar imágenes del conjunto de prueba (test)")
        print("2. Usar la cámara en tiempo real")
        print("3. Seleccionar una imagen del dispositivo")
        print("4. Salir")
        choice = input("Seleccione una opción: ").strip()

        if choice == '1':
            test_images()
        elif choice == '2':
            use_camera()
        elif choice == '3':
            process_image()
        elif choice == '4':
            print("[INFO] Saliendo del programa. ¡Adiós!")
            break
        else:
            print("[ERROR] Opción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main_menu()
