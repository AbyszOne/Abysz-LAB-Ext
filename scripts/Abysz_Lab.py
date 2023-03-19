import gradio as gr
import subprocess
import os
import imageio
from gradio.outputs import Image
from PIL import Image
import sys
import cv2
import shutil
import time

from modules import shared
from modules import scripts
from modules import script_callbacks


class Script(scripts.Script):
    def title(self):
        return "Abysz LAB"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
     
    def ui(self, is_img2img):
        return []
        
def main(ruta_entrada_1, ruta_entrada_2, ruta_salida, frames_limit, denoise_blur, dfi_strength, frame_refresh_frequency, refresh_strength, smooth, dfi_deghost, ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength, norm_strength):
            
        maskD = os.path.join(os.getcwd(), 'extensions', 'Abysz-LAB-Ext', 'scripts', 'Run', 'MaskD')
        maskS = os.path.join(os.getcwd(), 'extensions', 'Abysz-LAB-Ext', 'scripts', 'Run', 'MaskS')
        #output = os.path.join(os.getcwd(), 'extensions', 'Abysz-lab', 'scripts', 'Run', 'Output')
        source = os.path.join(os.getcwd(), 'extensions', 'Abysz-LAB-Ext', 'scripts', 'Run', 'Source')
        #gen = os.path.join(os.getcwd(), 'extensions', 'Abysz-lab', 'scripts', 'Run', 'Gen')

        
        os.makedirs(source, exist_ok=True)
        os.makedirs(maskS, exist_ok=True)
        os.makedirs(ruta_salida, exist_ok=True)
        os.makedirs(maskD, exist_ok=True)
        #os.makedirs(gen, exist_ok=True)
        
        # Copy the images from Ruta1 to Source folder in JPEG quality 100
        #for file in os.listdir(ruta_entrada_1):
        #    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        #        img = Image.open(os.path.join(ruta_entrada_1, file))
        #        img.save(os.path.join("Source", file), "jpeg", quality=100)
        def copy_images(ruta_entrada_1, ruta_entrada_2, frames_limit=0):
            # Copiar todas las imágenes de la carpeta ruta_entrada_1 a la carpeta Source
            count = 0
            for file in os.listdir(ruta_entrada_1):
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    img = Image.open(os.path.join(ruta_entrada_1, file))
                    rgb_img = img.convert('RGB')
                    rgb_img.save(os.path.join("./extensions/Abysz-LAB-Ext/scripts/Run/Source", file), "jpeg", quality=100)
                    count += 1
                    if frames_limit > 0 and count >= frames_limit:
                        break
        
        # Llamar a la función copy_images para copiar las imágenes
        copy_images(ruta_entrada_1,ruta_salida, frames_limit)
        
        #for file in os.listdir(ruta_entrada_2):
        #    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
        #        img = Image.open(os.path.join(ruta_entrada_2, file))
        #        img.save(os.path.join(ruta_entrada_2, file))
        #        
        # Carpeta donde se encuentran las imágenes de Gen
        def sresize(ruta_entrada_2):
            gen_folder = ruta_entrada_2
            
            # Carpeta donde se encuentran las imágenes de FULL
            full_folder = "./extensions/Abysz-LAB-Ext/scripts/Run/Source"
            
            # Obtener la primera imagen en la carpeta Gen
            gen_images = os.listdir(gen_folder)
            gen_image_path = os.path.join(gen_folder, gen_images[0])
            gen_image = cv2.imread(gen_image_path)
            gen_height, gen_width = gen_image.shape[:2]
            gen_aspect_ratio = gen_width / gen_height
            
            # Recorrer todas las imágenes en la carpeta FULL
            for image_name in os.listdir(full_folder):
                image_path = os.path.join(full_folder, image_name)
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                aspect_ratio = width / height
            
                if aspect_ratio != gen_aspect_ratio:
                    if aspect_ratio > gen_aspect_ratio:
                        # La imagen es más ancha que la imagen de Gen
                        crop_width = int(height * gen_aspect_ratio)
                        x = int((width - crop_width) / 2)
                        image = image[:, x:x+crop_width]
                    else:
                        # La imagen es más alta que la imagen de Gen
                        crop_height = int(width / gen_aspect_ratio)
                        y = int((height - crop_height) / 2)
                        image = image[y:y+crop_height, :]
            
                # Redimensionar la imagen de FULL a la resolución de la imagen de Gen
                image = cv2.resize(image, (gen_width, gen_height))
            
                # Guardar la imagen redimensionada en la carpeta FULL
                cv2.imwrite(os.path.join(full_folder, image_name), image)
        
        sresize(ruta_entrada_2)
            
        def s_g_rename(ruta_entrada_2):
            source_dir = "./extensions/Abysz-LAB-Ext/scripts/Run/Source" # ruta de la carpeta "Source"
            
            # Obtener una lista de los nombres de archivo en la carpeta "Source"
            files = os.listdir(source_dir)
            
            # Renombrar cada archivo
            for i, file_name in enumerate(files):
                old_path = os.path.join(source_dir, file_name) # ruta actual del archivo
                new_file_name = f"{i+1:03d}" # nuevo nombre de archivo con formato %03d
                new_path = os.path.join(source_dir, new_file_name + os.path.splitext(file_name)[1]) # nueva ruta del archivo
                try:
                    os.rename(old_path, new_path)
                except FileExistsError:
                    print(f"El archivo {new_file_name} ya existe. Se omite su renombre.")
                
            gen_dir = ruta_entrada_2 # ruta de la carpeta "Source"
            
            # Obtener una lista de los nombres de archivo en la carpeta ruta_entrada_2
            files = os.listdir(gen_dir)
            
            # Renombrar cada archivo
            for i, file_name in enumerate(files):
                old_path = os.path.join(gen_dir, file_name) # ruta actual del archivo
                new_file_name = f"{i+1:03d}" # nuevo nombre de archivo con formato %03d
                new_path = os.path.join(gen_dir, new_file_name + os.path.splitext(file_name)[1]) # nueva ruta del archivo
                try:
                    os.rename(old_path, new_path)
                except FileExistsError:
                    print(f"El archivo {new_file_name} ya existe. Se omite su renombre.")
        
        s_g_rename(ruta_entrada_2)
        
        # Obtener el primer archivo de la carpeta ruta_entrada_2
        gen_files = os.listdir(ruta_entrada_2)
        if gen_files:
            first_gen_file = gen_files[0]

            # Copiar el archivo a la carpeta "Output" y reemplazar si ya existe
            #output_file = "Output" + first_gen_file
            #shutil.copyfile(ruta_entrada_2 + first_gen_file, output_file)
            output_file = os.path.join(ruta_salida, first_gen_file)
            shutil.copyfile(os.path.join(ruta_entrada_2, first_gen_file), output_file)
        #subprocess call
        def denoise(denoise_blur):
            denoise_kernel = denoise_blur
            # Obtener la lista de nombres de archivos en la carpeta source
            files = os.listdir("./extensions/Abysz-LAB-Ext/scripts/Run/Source")
            
            # Crear una carpeta destino si no existe
            #if not os.path.exists("dest"):
            #   os.mkdir("dest")
            
            # Recorrer cada archivo en la carpeta source
            for file in files:
                # Leer la imagen con opencv
                img = cv2.imread(os.path.join("./extensions/Abysz-LAB-Ext/scripts/Run/Source", file))
            
                # Aplicar el filtro de blur con un tamaño de kernel 5x5
                dst = cv2.blur(img, (denoise_kernel, denoise_kernel))
                
                # Eliminar el archivo original
                #os.remove(os.path.join("SourceDFI", file))
            
                # Guardar la imagen resultante en la carpeta destino con el mismo nombre
                cv2.imwrite(os.path.join("./extensions/Abysz-LAB-Ext/scripts/Run/Source", file), dst)
                
        denoise(denoise_blur)    
        
        # Definir la carpeta donde están los archivos
        carpeta = './extensions/Abysz-LAB-Ext/scripts/Run/Source'
        
        # Crear la carpeta MaskD si no existe
        os.makedirs('./extensions/Abysz-LAB-Ext/scripts/Run/MaskD', exist_ok=True)
        
        # Inicializar contador
        contador = 1
        
        umbral_size = dfi_strength
        # Iterar a través de los archivos de imagen en la carpeta Source
        for filename in sorted(os.listdir(carpeta)):
            # Cargar la imagen actual y la siguiente en escala de grises
            if contador > 1:
                siguiente = cv2.imread(os.path.join(carpeta, filename), cv2.IMREAD_GRAYSCALE)
                diff = cv2.absdiff(anterior, siguiente)
        
                # Aplicar un umbral y guardar la imagen resultante en la carpeta MaskD. Menos es más.
                umbral = umbral_size
                umbralizado = cv2.threshold(diff, umbral, 255, cv2.THRESH_BINARY_INV)[1] # Invertir los colores
                cv2.imwrite(os.path.join('./extensions/Abysz-LAB-Ext/scripts/Run/MaskD', f'{contador-1:03d}.png'), umbralizado)
        
            anterior = cv2.imread(os.path.join(carpeta, filename), cv2.IMREAD_GRAYSCALE)
            contador += 1
            
            #Actualmente, el tipo de umbralización es cv2.THRESH_BINARY_INV, que invierte los colores de la imagen umbralizada. 
            #Puedes cambiarlo a otro tipo de umbralización, 
            #como cv2.THRESH_BINARY, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO o cv2.THRESH_TOZERO_INV.
        
        
        # Obtener la lista de los nombres de los archivos en la carpeta MaskD
        files = os.listdir("./extensions/Abysz-LAB-Ext/scripts/Run/MaskD")
        # Definir la carpeta donde están los archivos
        carpeta = "./extensions/Abysz-LAB-Ext/scripts/Run/MaskD"
        blur_kernel = smooth
        
        # Iterar sobre cada archivo
        for file in files:
            if dfi_deghost == 0:
                
                continue
            # Leer la imagen de la carpeta MaskD
            #img = cv2.imread("MaskD" + file)
            img = cv2.imread(os.path.join("./extensions/Abysz-LAB-Ext/scripts/Run/MaskD", file))
            
            # Invertir la imagen usando la función bitwise_not()
            img_inv = cv2.bitwise_not(img)
            
            kernel_size = dfi_deghost
            
            # Dilatar la imagen usando la función dilate()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)) # Puedes cambiar el tamaño y la forma del kernel según tus preferencias
            img_dil = cv2.dilate(img_inv, kernel)
            
            # Volver a invertir la imagen usando la función bitwise_not()
            img_out = cv2.bitwise_not(img_dil)
            
            # Sobrescribir la imagen en la carpeta MaskD con el mismo nombre que el original
            #cv2.imwrite("MaskD" + file, img_out)
            #cv2.imwrite(os.path.join("MaskD", file, img_out))
            filename = os.path.join("./extensions/Abysz-LAB-Ext/scripts/Run/MaskD", file)
            cv2.imwrite(filename, img_out)
    
        # Iterar a través de los archivos de imagen en la carpeta MaskD
        for imagen in os.listdir(carpeta):
            if imagen.endswith(".jpg") or imagen.endswith(".png") or imagen.endswith(".jpeg"):
                # Leer la imagen
                img = cv2.imread(os.path.join(carpeta, imagen))
                # Aplicar el filtro
                img = cv2.GaussianBlur(img, (blur_kernel,blur_kernel),0)
                # Guardar la imagen con el mismo nombre
                cv2.imwrite(os.path.join(carpeta, imagen), img)
        
        
        # INICIO DEL BATCH Obtener el nombre del archivo en MaskD sin ninguna extensión
        # Agregar una variable de contador de bucles
        loop_count = 0
        
        # Agregar un bucle while para ejecutar el código en bucle infinito
        while True:
        
            mask_files = sorted(os.listdir(maskD))
            if not mask_files:
                print(f"No frames left")
                # Eliminar las carpetas Source, MaskS y MaskD si no hay más archivos para procesar
                shutil.rmtree(maskD)
                shutil.rmtree(maskS)
                shutil.rmtree(source)
                break
                
            mask = mask_files[0]
            maskname = os.path.splitext(mask)[0]
            
            # Obtener la ruta de la imagen en la subcarpeta de output que tiene el mismo nombre que la imagen en MaskD
            output_files = [f for f in os.listdir(ruta_salida) if os.path.splitext(f)[0] == maskname]
            if not output_files:
                print(f"No se encontró en {ruta_salida} una imagen con el mismo nombre que {maskname}.")
                exit(1)
            
            output_file = os.path.join(ruta_salida, output_files[0])
            
            # Aplicar el comando magick composite con las opciones deseadas
            composite_command = f"magick composite -compose CopyOpacity {os.path.join(maskD, mask)} {output_file} {os.path.join(maskS, 'result.png')}"
            os.system(composite_command)
            
            # Obtener el nombre del archivo en output sin ninguna extensión
            name = os.path.splitext(os.path.basename(output_file))[0]
            
            # Renombrar el archivo result.png con el nombre del archivo en output y la extensión .png
            os.rename(os.path.join(maskS, 'result.png'), os.path.join(maskS, f"{name}.png"))
            
            #Guardar el directorio actual en una variable
            original_dir = os.getcwd()
            
            #Cambiar al directorio de la carpeta MaskS
            os.chdir(maskS)
            
            #Iterar a través de los archivos de imagen en la carpeta MaskS
            for imagen in sorted(os.listdir(".")):
                # Obtener el nombre de la imagen sin la extensión
                nombre, extension = os.path.splitext(imagen)
                # Obtener solo el número de la imagen
                numero = ''.join(filter(str.isdigit, nombre))
                # Definir el nombre de la siguiente imagen
                siguiente = f"{int(numero)+1:0{len(numero)}}{extension}"
                # Renombrar la imagen
                os.rename(imagen, siguiente)
            
            # Volver al directorio original
            os.chdir(original_dir)
            
            # Establecer un valor predeterminado para disolución
            dissolve = 100 if loop_count % frame_refresh_frequency != 0 else refresh_strength
            #slider2 = gr.inputs.Slider(minimum=0, maximum=100, default=50, step=5, label="FPR Strength")
        
        
            # Obtener el nombre del archivo en MaskS sin la extensión
            maskS_files = [f for f in os.listdir(maskS) if os.path.isfile(os.path.join(maskS, f)) and f.endswith('.png')]
            if maskS_files:
                filename = os.path.splitext(maskS_files[0])[0]
            else:
                print(f"No se encontraron archivos de imagen en la carpeta '{maskS}'")
                filename = ''[0]
            
            # Salir del bucle si no hay más imágenes que procesar
            if not filename:
                break
            
            # Obtener la extensión del archivo en Gen con el mismo nombre
            gen_files = [f for f in os.listdir(ruta_entrada_2) if os.path.isfile(os.path.join(ruta_entrada_2, f)) and f.startswith(filename)]
            if gen_files:
                ext = os.path.splitext(gen_files[0])[1]
            else:
                print(f"No se encontró ningún archivo con el nombre '{filename}' en la carpeta '{ruta_entrada_2}'")
                ext = ''
                                
            # Componer la imagen de MaskS y Gen con disolución (si está definido) y guardarla en la carpeta de salida
            os.system(f"magick composite {'-dissolve ' + str(dissolve) + '%' if dissolve is not None else ''} {maskS}/{filename}.png {ruta_entrada_2}/{filename}{ext} {ruta_salida}/{filename}{ext}")
            
            # Obtener el nombre del archivo más bajo en la carpeta MaskD
            maskd_files = [f for f in os.listdir(maskD) if os.path.isfile(os.path.join(maskD, f)) and f.startswith('')]
            if maskd_files:
                maskd_file = os.path.join(maskD, sorted(maskd_files)[0])
                os.remove(maskd_file)
            
            # Obtener el nombre del archivo más bajo en la carpeta MaskS
            masks_files = [f for f in os.listdir(maskS) if os.path.isfile(os.path.join(maskS, f)) and f.startswith('')]
            if masks_files:
                masks_file = os.path.join(maskS, sorted(masks_files)[0])
                os.remove(masks_file)
                                
            # Aumentar el contador de bucles
            loop_count += 1
            
 
def dyndef(ruta_entrada_3, ruta_salida_1, ddf_strength):
    if ddf_strength <= 0: # Condición 1: strength debe ser mayor a 0
        return
    imgs = []
    files = os.listdir(ruta_entrada_3)
    
    for file in files:
        img = cv2.imread(os.path.join(ruta_entrada_3, file))
        imgs.append(img)
    
    for idx in range(len(imgs)-1, 0, -1):
        current_img = imgs[idx]
        prev_img = imgs[idx-1]
        alpha = ddf_strength
        
        current_img = cv2.addWeighted(current_img, alpha, prev_img, 1-alpha, 0)
        imgs[idx] = current_img
        
        if not os.path.exists(ruta_salida_1):
            os.makedirs(ruta_salida_1)
            
        output_path = os.path.join(ruta_salida_1, files[idx]) # Usa el mismo nombre que el original
        cv2.imwrite(output_path, current_img)
    
    # Copia el primer archivo de los originales al finalizar el proceso
    shutil.copy(os.path.join(ruta_entrada_3, files[0]), os.path.join(ruta_salida_1, files[0]))



def overlay_images(image1_path, image2_path, over_strength):
      
    opacity = over_strength
    
    # Abrir las imágenes
    image1 = Image.open(image1_path).convert('RGBA')
    image2 = Image.open(image2_path).convert('RGBA')

    # Alinear el tamaño de las imágenes
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Convertir las imágenes en matrices NumPy
    np_image1 = np.array(image1).astype(np.float64) / 255.0
    np_image2 = np.array(image2).astype(np.float64) / 255.0

    # Aplicar el método de fusión "overlay" a las imágenes
    def basic(target, blend, opacity):
        return target * opacity + blend * (1-opacity)

    def blender(func):
        def blend(target, blend, opacity=1, *args):
            res = func(target, blend, *args)
            res = basic(res, blend, opacity)
            return np.clip(res, 0, 1)
        return blend

    class Blend:
        @classmethod
        def method(cls, name):
            return getattr(cls, name)

        normal = basic

        @staticmethod
        @blender
        def overlay(target, blend, *args):
            return  (target>0.5) * (1-(2-2*target)*(1-blend)) +\
                    (target<=0.5) * (2*target*blend)

    blended_image = Blend.overlay(np_image1, np_image2, opacity)

    # Convertir la matriz de vuelta a una imagen PIL
    blended_image = Image.fromarray((blended_image * 255).astype(np.uint8), 'RGBA').convert('RGB')

    # Guardar la imagen resultante
    return blended_image

def overlay_run(ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength):
    if over_strength <= 0: # Condición 1: strength debe ser mayor a 0
            return
       
    # Si ddf_strength y/o over_strength son mayores a 0, utilizar ruta_salida_1 en lugar de ruta_entrada_3
    if ddf_strength > 0:
        ruta_entrada_3 = ruta_salida_1    
        
    if not os.path.exists("overtemp"):
            os.makedirs("overtemp")
            
    gen_path = ruta_entrada_3
    images = os.listdir(gen_path)
    image1_path = os.path.join(gen_path, images[0])
    image2_path = os.path.join(gen_path, images[1])
    
     
    fused_image = overlay_images(image1_path, image2_path, over_strength)
    fuseover_path = "overtemp"
    filename = os.path.basename(image1_path)
    fused_image.save(os.path.join(fuseover_path, filename))
    
    
    # Obtener una lista de todos los archivos en la carpeta "Gen"
    gen_files = os.listdir(ruta_entrada_3)
    
    for i in range(len(gen_files) - 1):
        image1_path = os.path.join(ruta_entrada_3, gen_files[i])
        image2_path = os.path.join(ruta_entrada_3, gen_files[i+1])
        blended_image = overlay_images(image1_path, image2_path, over_strength)
        blended_image.save(os.path.join("overtemp", gen_files[i+1]))
    
    
    # Definimos la ruta de la carpeta "overtemp"
    ruta_overtemp = "overtemp"
    
    # Movemos todos los archivos de la carpeta "overtemp" a la carpeta "ruta_salida"
    for archivo in os.listdir(ruta_overtemp):
        origen = os.path.join(ruta_overtemp, archivo)
        destino = os.path.join(ruta_salida_1, archivo)
        shutil.move(origen, destino)


def norm(ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength, norm_strength):
    if norm_strength <= 0: # Condición 1: Norm_strength debe ser mayor a 0
        return
    
    # Si ddf_strength y/o over_strength son mayores a 0, utilizar ruta_salida_1 en lugar de ruta_entrada_3
    if ddf_strength > 0 or over_strength > 0:
        ruta_entrada_3 = ruta_salida_1
    
    # Crear la carpeta GenOverNorm si no existe
    if not os.path.exists("normtemp"):
        os.makedirs("normtemp")
        
    # Obtener una lista de todas las imágenes en la carpeta FuseOver
    img_list = os.listdir(ruta_entrada_3)
    img_list.sort() # Ordenar la lista en orden ascendente
        
    # Iterar a través de las imágenes
    for i in range(len(img_list)-1):
        # Cargar las dos imágenes a fusionar
        img1 = cv2.imread(os.path.join(ruta_entrada_3, img_list[i]))
        img2 = cv2.imread(os.path.join(ruta_entrada_3, img_list[i+1]))
    
        # Calcular la luminosidad promedio de cada imagen
        avg1 = np.mean(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        avg2 = np.mean(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    
        # Calcular los pesos para cada imagen
        weight1 = avg1 / (avg1 + avg2)
        weight2 = avg2 / (avg1 + avg2)
    
        # Fusionar las imágenes utilizando los pesos
        result = cv2.addWeighted(img1, weight1, img2, weight2, 0)
                                
        # Guardar la imagen resultante en la carpeta GenOverNorm con el mismo nombre que la imagen original
        cv2.imwrite(os.path.join("normtemp", img_list[i+1]), result)
    
    # Copiar la primera imagen en la carpeta GenOverNorm para mantener la secuencia completa
    img0 = cv2.imread(os.path.join(ruta_entrada_3, img_list[0]))
    cv2.imwrite(os.path.join("normtemp", img_list[0]), img0)
    
    # Definimos la ruta de la carpeta "overtemp"
    ruta_overtemp = "normtemp"
    
    # Movemos todos los archivos de la carpeta "overtemp" a la carpeta "ruta_salida"
    for archivo in os.listdir(ruta_overtemp):
        origen = os.path.join(ruta_overtemp, archivo)
        destino = os.path.join(ruta_salida_1, archivo)
        shutil.move(origen, destino)
        
def deflickers(ruta_entrada_1, ruta_entrada_2, ruta_salida, frames_limit, denoise_blur, dfi_strength, frame_refresh_frequency, refresh_strength, smooth, dfi_deghost, ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength, norm_strength):
    dyndef(ruta_entrada_3, ruta_salida_1, ddf_strength)
    overlay_run(ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength)
    norm(ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength, norm_strength)
    
def add_tab():
    print('LAB')
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Tabs():
            with gr.Tab("Settings"):
                with gr.Column():
                    gr.Markdown("# Abysz LAB 0.0.8. Temporal coherence tools")
                    gr.Markdown("## DFI Parameters")
                with gr.Row():
                    ruta_entrada_1 = gr.Textbox(label="Original frames folder", placeholder="Unless you have used --just resize-- with different aspect ratios, any source will work.")
                    ruta_entrada_2 = gr.Textbox(label="Generated frames folder", placeholder="The frames of AI generated video")
                ruta_salida = gr.Textbox(label="Output folder", placeholder="Remember that each generation overwrites previous frames in the same folder.")
                with gr.Accordion("Info", open=False):
                    gr.Markdown("**Source denoise:** The dynamic map scan is sensitive to noise. If your source is not very clean, you may want to add a soft blur to remove impurities. This does not modify the original.")
                    gr.Markdown("**DFI Strength:** This controls the sensitivity of the scan. For example, a low value will find too much detail even in apparently static areas, which is not recommended, and a high value will not detect minor movement.")
                    gr.Markdown("**Frame Refresh:** To control corruption, inevitable in such an aggressive process, you can refresh a frame every X cycles.")
                    gr.Markdown("**Refresh Control:** Here you decide how much interpolation % you allow in each refresh. The lower, the more similar it will be to the original frame. This will refresh the corruption, but it will also allow flicking.")
                    gr.Markdown("**Smooth:** This smoothes the edges of the interpolated areas. Low values are currently recommended until the algorithm is updated.")
                    gr.Markdown("**DFI Deghost:** This artificially fattens the edges of the areas to be interpolated. This can be useful if you don't want to change DFI Strength but need to reduce ghosting. Experimental. 0=Off.")
                with gr.Row():
                    frames_limit = gr.Number(label="Frames to render. 0:ALL")
                    denoise_blur = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Source denoise")
                    dfi_strength = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="DFI Strength")
                    frame_refresh_frequency = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Frame Refresh")
                    refresh_strength = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Refresh Control")
                    smooth = gr.Slider(minimum=1, maximum=99, value=11, step=2, label="Smooth")
                    dfi_deghost = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="DFI Deghost")
                with gr.Row():
                    run_button = gr.Button(label="Run", variant="primary")
                    output_placeholder = gr.Textbox(label="Status")
                with gr.Column():
                    gr.Markdown("## Deflickers playground")
                    with gr.Accordion("Info", open=False):
                        gr.Markdown("I made this series of deflickers based on the standard that Vegas Pro includes. You can use them together or separately. Be careful when mixing them.")
                        gr.Markdown("**Blend:** Blends a percentage of the previous frame with the next. This can soften transitions and highlights. High values will add too much information.")
                        gr.Markdown("**Overlay:** Use the overlay image blending mode. Although it can be effective in reducing changes, note that this will modify the contrast. You will have to decide if it works for you.")
                        gr.Markdown("**Normalize:** Calculates the average between frames to merge them. It may be more practical if you don't have a specific Blend deflicker value in mind.")
                with gr.Row():
                    with gr.Column():
                        ruta_entrada_3 = gr.Textbox(label="Frames folder", placeholder="Frames to process")
                        ruta_salida_1 = gr.Textbox(label="Output folder", placeholder="Processed frames")
                        dfk_button = gr.Button(label="Run")
                    with gr.Column():
                        ddf_strength = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label="BLEND (0=Off)")
                        over_strength = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label="OVERLAY (0=Off)")
                        norm_strength = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="NORMALIZE (0=Off))")
                    
            with gr.Tab("Lab Tools"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("# Coming Soon")
            with gr.Tab("Guide"):
                with gr.Column():
                    gr.Markdown("# What DFI does?")
                    with gr.Accordion("Info", open=False):
                        gr.Markdown("DFI processing analyzes the motion of the original video, and attempts to force that information into the generated video. (Demo on github page)")
                        gr.Markdown("For example, for a man smoking, leaning against a pole, it will detect that the pole is static, and will try to prevent it from changing as much as possible.")
                        gr.Markdown("This is an aggressive process that requires a lot of control for each context. Read the recommended strategies in the manual: https://github.com/AbyszOne/Abysz_lab")
                        gr.Markdown("Although Video to Video is the most efficient way, a DFI One Shot method is under experimental development as well.")
        inputs=[ruta_entrada_1, ruta_entrada_2, ruta_salida, frames_limit, denoise_blur, dfi_strength, frame_refresh_frequency, refresh_strength, smooth, dfi_deghost, ruta_entrada_3, ruta_salida_1, ddf_strength, over_strength, norm_strength]
        run_button.click(fn=main, inputs=inputs, outputs=output_placeholder)
        dfk_button.click(fn=deflickers, inputs=inputs, outputs=output_placeholder)
    return [(demo, "Abysz LAB", "demo")]
        
script_callbacks.on_ui_tabs(add_tab)
