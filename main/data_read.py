import os
import sys, traceback
import random

import cv2 as cv
from preprare_image import PrepareImage

class DataRead:

	def __init__(self):
		self._labels = {'hombre': 0, 'mujer': 1}
		self._inferencia_log_file = "inferencia_log.txt"


	def get_label(self, class_name):
		return self._labels[class_name]


	def count_files(self, source_dirname):
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
		for tf_file in os.listdir(os.path.dirname(source_dirname)):
			if tf_file.lower().endswith('.tfrecord'):
				number_of_files += 1
		return number_of_files


	def read_byname(self, source_dirname):
		"""
		Provee a la red un método de entrada para la predicción en imagenes no vistas
		NOTA: la diferencia de read_file es que este método genera las etiquetas para cada
		imagen basandose en el nombre de la imagen el cual debe estar especificado de la
		forma: mujer_0.jpg, hombre_0.jpg. Siendo '_' el delimitador que separa la clase de
		la imagen del resto del nombre de cada imagen.
		"""
		absolute_path_images = []
		labels = []
		try:
			for image in os.listdir(source_dirname):
				labels.append(self.get_label(os.path.basename(image).split('_')[0]))
				absolute_path_images.append(os.path.join(source_dirname, image))

			return absolute_path_images, labels
		except IOError:
			print("read_byname_files:Error-->{0}".format(sys.exc_info()))


	def read_bydirectories(self, source_dirname):
		"""
		Proporciona un método de entrada para la predicción de imagenes no vistas.
		las imagenes deben encontrarse con la siguiente estructura:
		directorio-
			..directorio_clase_1/imagen_x.jpg
			..directorio_clase_2/imagen_x.jpg
		"""
		try:
			absolute_path_images = []
			labels = []
			for classes in os.listdir(source_dirname):
				if len(classes) > 0:
					class_name = classes.strip('\\')
					source_dirname_gender = os.path.join(source_dirname, class_name)
					for image in os.listdir(source_dirname_gender):
						absolute_path_images.append(os.path.join(source_dirname_gender, image))
						labels.append(self.get_label(class_name))

			return absolute_path_images, labels
		except IOError:
			print("read_bydirectorioes:Error-->{0}".format(sys.exc_info()))


	def read_byrandom(self, source_dirname):
		"""
		Proporciona un método de entrada para la predicción de imágenes no vistas.
		Este método genera las etiquetas para cada imagen con valores aleatorios de 0(hombre) a
		1(mujer) asumiendo una implementación del mundo real donde no se sabe a que clase pertenece
		la imagen a predecir.
		Args:
			source_dirname: string; ruta en crudo de la carpeta contenedora de las imagenes para realizar
			la inferencia.
		Return:
			absolute_path_images: lista; rutas de cada imagen.
			labels: lista; etiquetas de las imagenes generadas aleatoriamente.
		"""
		absolute_path_images = []
		labels = []
		for image in os.listdir(source_dirname):
			absolute_path_images.append(os.path.join(source_dirname, image))
			labels.append(random.randrange(0, 2))
		return absolute_path_images, labels


	def save_image(self, full_image_path, predicted_label, image_index):
		"""
		Guarda la imagen de acuerdo a la predicción realizada(gender) en la carpeta de su clase
		correspondiente: data/mujer/..  | data/hombre/..
		Args:
			full_img_path: string; ruta en crudo de la imagen.
			predicted_label: int; clase inferida por el modelo 0|1.
			image_index: int; número de imagen a la que se le realizó inferencia.
		"""
		try:
			image_name = os.path.basename(full_image_path)
			cv_image = cv.imread(full_image_path, 0)
			inference_dir_path = "/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/inferencia/"

			if predicted_label == 1:
				inference_directory_fullpath = create_dirs(inference_dir_path, "mujer")
				new_image_name = "mujer_"+str(image_index)+".jpg"
				cv.imwrite(os.path.join(inference_directory_fullpath, new_image_name), cv_image)
				PrepareImage().writeOnFile(image_name, new_image_name, self._inferencia_log_file)
			elif predicted_label == 0:
				inference_directory_fullpath = create_dirs(inference_dir_path, "hombre")
				new_image_name = "hombre_"+str(image_index)+".jpg"
				cv.imwrite(os.path.join(inference_directory_fullpath, new_image_image), cv_image)
				PrepareImage().writeOnFile(image_name, new_image_image, self._inferencia_log_file)
		except IOError:
			print("Error-->{0}".format(sys.exc_info()))


	def create_dirs(self, directory_path, gender):
		gender_directory_fullpath = os.path.join(directory_path, gender)
		if not os.path.exists(gender_directory_fullpath):
			os.makedirs(gender_directory_fullpath)

		return gender_directory_fullpath
