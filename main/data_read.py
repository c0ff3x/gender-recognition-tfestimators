import cv2 as cv
import os
import sys, traceback
import random
from preprare_image import *

labels = {'hombre': 0, 'mujer': 1}


def get_label(class_name):
	return labels[class_name]


def count_files(source_dirname):
	"""
	Cuenta el número de elementos tfrecord que se encuentran dentro de source_dirname,
	asume que todos los elementos dentro de la carpeta están en dicho formato.
	(slpitext return a tuple with root and extention file type)
	Args:
		source_dirname: string, ruta en crudo de la carpeta contenedora de los archivos tfrecord.
	Return:
		int: número de archvios dentro de la carpeta.
	"""
	number_of_files = 0
	for file in os.listdir(os.path.dirname(source_dirname)):
		if os.path.splitext(file)[1].lower() == ".tfrecord":
			number_of_files += 1
	return number_of_files


def read_byname(source_dirname):
	"""
	Provee a la red un método de entrada para la predicción en imagenes no vistas
	NOTA: la diferencia de read_file es que este método genera las etiquetas para cada
	imagen basandose en el nombre de la imagen el cual debe estar especificado de la
	forma: mujer_0.jpg, hombre_0.jpg. Siendo '_' el delimitador que separa la clase de
	la imagen del resto del nombre de cada imagen.
	"""
	abs_path_images = []
	label = []
	try:
		for image in os.listdir(source_dirname):
			basename = os.path.basename(image).split('_')[0]
			label.append(get_label(basename))
			abs_path_images.append(os.path.join(source_dirname, image))

		return abs_path_images, label
	except IOError:
		print("read_byname_files:Error-->{0}".format(sys.exc_info()))


def read_bydirectories(source_dirname):
	"""
	Proporciona un método de entrada para la predicción de imagenes no vistas.
	las imagenes deben encontrarse con la siguiente estructura:
	directorio-
		..directorio_clase_1/imagen_x.jpg
		..directorio_clase_2/imagen_x.jpg
	"""
	try:
		abs_path_images = []
		label = []
		for classes in os.listdir(source_dirname):
			if len(classes) > 0:
				class_name = classes.strip('\\')
				source_dirname_gender = os.path.join(source_dirname, class_name)
				for image in os.listdir(source_dirname_gender):
					abs_path_images.append(os.path.join(source_dirname_gender, image))
					label.append(get_label(class_name))

		return abs_path_images, label
	except IOError:
		print("read_bydirectorioes:Error-->{0}".format(sys.exc_info()))


def read_byrandom(source_dirname):
	"""
	Proporciona un método de entrada para la predicción de imágenes no vistas.
	Este método genera las etiquetas para cada imagen con valores aleatorios de 0(hombre) a
	1(mujer) asumiendo una implementación del mundo real donde no se sabe a que clase pertenece
	la imagen a predecir.
	Args:
		source_dirname: string; ruta en crudo de la carpeta contenedora de las imagenes para realizar
		la inferencia.
	Return:
		abs_path_images: lista; rutas de cada imagen.
		label: lista; etiquetas de las imagenes generadas aleatoriamente.
	"""
	abs_path_images = []
	label = []
	for img in os.listdir(source_dirname):
		abs_path_images.append(os.path.join(source_dirname, img))
		label.append(random.randrange(0, 2))
	return abs_path_images, label


def save_image(full_img_path, predicted_label, indice):
	"""
	Guarda la imagen de acuerdo a la predicción realizada(gender) en la carpeta de su clase
	correspondiente: data/mujer/..  | data/hombre/..
	Args:
		full_img_path: string; ruta en crudo de la imagen.
		predicted_label: int; clase inferida por el modelo 0|1.
		indice: int; número de imagen a la que se le realizó inferencia.
	"""
	try:
		image_name = os.path.basename(full_img_path)
		cv_image = cv.imread(full_img_path, 0)
		path = "/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/inferencia/"

		if predicted_label == 1:
			full_path = create_dirs(path, "mujer")
			new_image_name = "mujer_"+str(indice)+".jpg"
			cv.imwrite(os.path.join(full_path, new_image_name), cv_image)
			PrepareImage().writeOnFile(image_name, new_image_name, "inferencia_log.txt")
		elif predicted_label == 0:
			full_path = create_dirs(path, "hombre")
			new_image_name = "hombre_"+str(indice)+".jpg"
			cv.imwrite(os.path.join(full_path, new_image_image), cv_image)
			PrepareImage().writeOnFile(image_name, new_image_image, "inferencia_log.txt")
	except IOError:
		print("Error-->{0}".format(sys.exc_info()))


def create_dirs(path, gender):
	full_path = os.path.join(path, gender)
	if not os.path.exists(full_path):
		os.makedirs(full_path)

	return full_path
