import os
import subprocess
import time

def train_yolov4():
    """
    Función para entrenar el modelo de detección YOLOv4 en Darknet, para diferenciar entre hombres y mujeres.
    Incluye manejo de errores y monitoreo del entrenamiento.
    """
    # Define las rutas a los archivos necesarios
    darknet_path = r"C:\Users\ferik\Desktop\darknet-master"  # Ajusta esta ruta a tu instalación de Darknet
    cfg_path = r"C:\Users\ferik\Desktop\ProyectoU4\cfg\yolov4-obj.cfg"  # Archivo de configuración de YOLOv4
    obj_data_path = r"C:\Users\ferik\Desktop\ProyectoU4\data\obj.data"  # Archivo de datos (obj.data)
    obj_name_path = r"C:\Users\ferik\Desktop\ProyectoU4\data\obj.name"  # Archivo de clases (obj.name)
    weights_path = r"C:\Users\ferik\Desktop\ProyectoU4\backup"  # Carpeta donde se guardarán los pesos
    pretrained_weights = r"C:\Users\ferik\Desktop\ProyectoU4\yolov4.weights"  # Pesos preentrenados (si los estás usando)

    # Comando para entrenar el modelo de YOLOv4
    command = [
        "darknet", "detector", "train", 
        obj_data_path, 
        cfg_path, 
        pretrained_weights,  # Utiliza los pesos preentrenados, si los tienes
        ]

    # Cambia al directorio de Darknet
    os.chdir(darknet_path)
    
    # Intenta ejecutar el comando y monitorear el progreso
    try:
        print("Iniciando el entrenamiento de YOLOv4...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Monitorear la salida en tiempo real
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Imprime las líneas de salida para ver el progreso

            # Puedes agregar un control de tiempo para interrumpir el proceso si dura demasiado
            # Esto es solo un ejemplo de cómo manejar el tiempo, puedes personalizarlo
            time.sleep(1)

        # Esperar que el proceso termine y obtener los posibles errores
        return_code = process.poll()
        if return_code != 0:
            stderr = process.stderr.read()
            raise Exception(f"Error en el entrenamiento: {stderr}")
        
        print("Entrenamiento completado exitosamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el entrenamiento: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    finally:
        print("Fin del proceso de entrenamiento.")

if __name__ == "__main__":
    train_yolov4()
