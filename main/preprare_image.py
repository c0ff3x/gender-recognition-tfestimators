"""
This code resizes the images for tfrecord files
"""
import os
import sys, traceback
import random
import shutil
import time

import cv2 as cv
import dlib
import numpy as np
from PIL import Image
from skimage import io


class PrepareImage(object):

	def __init__(self):
		"""
		PATH_ERROR_IMAGES: string; ruta del directorio de imagenes que contienen errores
		IMG_SIZE: int; tamaño al cual las imágenes serán redimensionadas"""
		self.__TRAIN_DIRECTORY = '/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/data/'	
		self.__TEST_DIRECTORY = '/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/testset/'
		self.__destination_dirname = os.path.dirname(os.path.realpath(__file__))+'/train/'
		self.__PATH_ERROR_IMAGES= os.path.dirname(os.path.realpath(__file__))+'/Error_Images/'
		self.__nofaces_dirname = os.path.dirname(os.path.realpath(__file__))+"/train_negatives/"
		self.__face_cascade = cv.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
		self.__IMG_SIZE = 128
		self.__allowed_image_extention = ["jpg", "jpeg", "png"]
		self.__full_image = []


	def __create_folder(self, directory):
		"""Crea un folder en la ruta especificada
		Args:
			directory: string; ruta de la carpeta a crear"""
		try:
			if not os.path.exists(directory):
				os.makedirs(directory)
		except OSError:
			print("Error al crear el directorio: {0}".format(directory))


	def __verify_image(self, img_file):
		"""verifica que una imagen no tiene errores y revisa que las
		imágenes estén en formato jpg o jpeg dicho formato dividido
		en nombre de la imagen y extensión en una lista
		[nombre_imagen, extensión]
		Args:
			img_file: string; ruta de la imagen a analizar"""
		try:
			img = io.imread(img_file)
			if os.path.basename(img_file).split(".")[1].lower() in self.__allowed_image_extention:
				#width, height
				width, height = img.shape[:2]
				if width > 20 and height > 20:
					return True
		except IndexError:
			print("Error: La imagen debe contener al menos un formato de archivo de imagen {0}".format(img_file))
			return False
		except Exception:
			return False


	def __is_gray_scale(self, image):
		"""Verifica si la imagen es Grayscale o RGB recorriendo los píxeles de
		la imagen y comparando si cada uno de cada nivel es igual R==G==B
		Args:
			image: string; ruta de la imagen a verificar."""
		try:
			img = Image.open(image).convert('RGB')
			width, height = img.size
			for i in range(width):
				for j in range(height):
					r, g, b = img.getpixel((i, j))
					if r != g != b: return False
			return True
		except:
			print("is_gray_scale: Error-->{0}".format(sys.exc_info()))


	def __get_correct_color_format(self, image):
		"""Dada una ruta de una imagen, carga una imagen con el objecto opencv de
		formato de color correcto usando el método is_gray_scale.
		Args:
			image: string, ruta en crudo de la imagen.
		Return:
			opencv_object, imagen con formato de color RGB o grayscale."""
		if self.__is_gray_scale(image):
			return cv.imread(image, 0)
		else:
#			cv_image = self.__histo_enhancement(cv_image)
			return cv.imread(image, 1)


	def __faceDetector(self, raw_image, class_name, destination_dirname, contador):
		"""Detecta los rostros dentro de una imagen
		Args:
			image: string; ruta en crudode la imagen
			classname: string, hombre|mujer:
			destination_dirname: string, carpeta donde se guardarán las imágenes:
			contador: integer, número de la imagen -> hombre_contador.jpg"""
		try:
#			descomentar si se desea re-dimensionar las imágenes para el posterior pre-porcesamiento
#			Esto fue realizado porque primero se redimensionan las imagenes para después ubicar el rostro
#			dentro de la imagen, esto decido de acuerdo al txt-Images_size_experiments vease -> pruebas de re-dimensionado
			img = self.__resize_keep_aspect_ratio(self.__get_correct_color_format(raw_image))
#			img = self.__get_correct_color_format(raw_image)
			faces = self.__face_cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 3, minSize=(20, 20))

			if len(faces) <= 0:
				return False

			for (x, y, w, h) in faces:
				#los rostros son detectados con x-y (esquina superior izquierda) y con el alto y ancho
				face_cuts = img[y:y + h, x:x + w]
				#si las imágenes que no están en blanco y negro se convierten
				if not self.__is_gray_scale(raw_image):
					face_cuts = cv.cvtColor(face_cuts, cv.COLOR_BGR2GRAY)
				self.__save_image(face_cuts, class_name, destination_dirname, contador)
			return True
		except:
			print("face_detector: Error al procesar la imagen, info--> {0}".format(sys.exc_info()))



	def __dlib_detector(self, raw_image, class_name, destination_dirname, contador):
		"""Detector de rostros usando la librería dlib,
		usado como un segundo mecanismo de detección de rostro si opencv
		face detector no detecta algún rostro dentro de la imagen
		Args:
			image: string; ruta en crudo de la imagen.
			class_name: string; hombre|mujer.
			destination_dirname: string; ruta donde serán guardadas las
			imágenes procesadas.
			contador: int; contador para nombre de las imágenes
			--> mujer_contador.jpg"""
		try:
#			descomentar si se desea re-dimensionar las imágenes para el posterior pre-porcesamiento
#			Esto fue realizado porque primero se redimensionan las imagenes para después ubicar el rostro
#			dentro de la imagen, esto decido de acuerdo al txt-Images_size_experiments vease -> pruebas de re-dimensionado
			img = self.__resize_keep_aspect_ratio(self.__get_correct_color_format(raw_image))
#			img = self.__get_correct_color_format(raw_image)

			face_detector = dlib.get_frontal_face_detector()
			detected_faces = face_detector(img, 1)

			if len(detected_faces) <= 0:
				return False
				"""if not self.__is_gray_scale(image):
					img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
				self.__save_image(img, class_name, destination_dirname, contador)"""

			for i, face_rect in enumerate(detected_faces):
				image_to_crop = Image.open(raw_image).convert('RGB')
				crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())

				cropped_image = image_to_crop.crop(crop_area)
				#el tamaño de una imagen esta dado de la forma height, width
				crop_size = (img.shape[0], img.shape[1])
				cropped_image.thumbnail(crop_size)
				#convierte la imagen de formato PIL.Image a opencv
				opencvImage = cv.cvtColor(np.array(cropped_image), cv.COLOR_RGB2GRAY)
				self.__save_image(opencvImage, class_name, destination_dirname, contador)
			return True

		except:
			print("Dlib_detector: Error al procesar la imagen, info -->{0}, {1}".format(sys.exc_info(), image))


	def __resize_keep_aspect_ratio(self, image):
		"""Redimensiona la imagen manteniendo su proporción de aspecto
		Args:
			image: opencv_object, imagen a redimensionar
		Return:
			new_im: opencv_object, imagen con las nuevas dimensiones
		"""
		try:
			#si no he leido la imagen con opencv descomentar esto
			#im = cv.imread(img, 1)
			old_size = image.shape[:2] #formato en (height, width)

			ratio = float(self.__IMG_SIZE)/max(old_size)
			new_size = ([int(x*ratio) for x in old_size])

			#new_size debe estar en (width, height)
			image = cv.resize(image, (new_size[1], new_size[0]))

			delta_w = self.__IMG_SIZE - new_size[1]
			delta_h = self.__IMG_SIZE - new_size[0]
			top, bottom = delta_h//2, delta_h - (delta_h//2)
			left, right = delta_w//2, delta_w - (delta_w//2)

			color = [0, 0, 0]
			new_im = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)
			return new_im
		except IOError:
			print("Resize_keep_aspect: Hubo errores al realizar las operaciones sobre la imagen {0} -->{1}".format(sys.exc_info(), image))


	def __histo_enhancement(self, img):
		"""Realiza Histogram equalization sobre una imagen a color.
		Args:
			img: opencv_object"""
		R, G, B = cv.split(img)
		output_R = cv.equalizeHist(R)
		output_G = cv.equalizeHist(G)
		output_B = cv.equalizeHist(B)
		output = cv.merge((output_R, output_G, output_B))
		return output


	def resize_images(self, source_dirname=None, destination_dirname=None):
		"""Dado un directorio con la forma
			../data/hombre/img-x.jpg
			../data/mujer/img-x.jpg
		procesa las imagenes para el archivo train-tfrecord y test-tfrecord
		cortando el rostro, redimensionando y cambiando su canal a 1
		Genera las carpetas:
			train/mujer/mujer-x.jpg
			train/hombre/hombre-x.jpg
			test/mujer/mujer-x.jpg
			test/hombre/hombre-x.jpg
		Args:
			source_dirname: string, carpeta donde se encuentran las imágenes
			a procesar
			destination_dirname: string, carpeta donde serán guardadas las
			imágenes procesadas
			"""
		start = time.time()
		source_dirname = self.__TRAIN_DIRECTORY if source_dirname is None else source_dirname
		destination_dirname = self.__destination_dirname if destination_dirname is None else destination_dirname
		print("Trabajando...")
		#recorremos la carpeta contenedora de las clases hombre/mujer
		for classes in os.listdir(source_dirname):
			if len(classes) > 0:
				contador = 0
				no_faces_counter = 0
				#dividimos para cada una de las clases y obtener la ruta de su
				#respectivo folder (mujer|hombre)
				class_name = classes.strip('\\')
#				creamos los registros para las imagenes procesadas con y sin rostros.
				self.writeOnFile(source_dirname, class_name, "imagenes_procesadas_data.txt")
				self.writeOnFile(source_dirname, class_name, "imagenes_sinrostros_data.txt")
				#nombre del folder donde se guardará las imágenes resultantes
				destination_dirname_gender = os.path.join(destination_dirname, class_name)
				#obtenemos la ruta final para cada clase: ./example/mujer-hombre
				source_dirname_gender = os.path.join(source_dirname, class_name)
				#iteramos dentro del folder de cada clase para obtener las imagenes
				for image in os.listdir(source_dirname_gender):
					#otiene la ruta completa de la imagen
					full_img_path = os.path.join(source_dirname_gender, image)
					#aqui podemos omititr el pasar la variable full_img_path
					#cv_image = cv.imread(full_img_path, 1)
					#llamada a método que captura los rostros
					if not self.__verify_image(full_img_path):
						print("la imagen contiene errores y será eliminada-descartada {0}".format(full_img_path))
						os.remove(full_img_path)
						continue

					if self.__faceDetector(full_img_path, class_name, destination_dirname_gender, contador):
						self.writeOnFile(os.path.basename(image), class_name+"_"+str(contador)+".jpg", "imagenes_procesadas_data.txt")
						contador += 1
					elif self.__dlib_detector(full_img_path, class_name, destination_dirname_gender, contador):
						self.writeOnFile(os.path.basename(image), class_name+"_"+str(contador)+".jpg", "imagenes_procesadas_data.txt")
						contador += 1
					else:
						nofaces_path = os.path.join(self.__nofaces_dirname, class_name)
						self.__create_folder(nofaces_path)
						self.writeOnFile(os.path.basename(image), os.path.basename(image), "imagenes_sinrostros_data.txt")
						shutil.move(full_img_path, nofaces_path)
						no_faces_counter += 1
				print("{0} imagenes sin rostros encontrados.".format(no_faces_counter))
			else:
				print("No se encotró ningún archivo en la ruta especificada")
		print("{0} imagenes sin rostros \n Hecho!".format(no_faces_counter))
		end = time.time()
		print(end-start)


	def resize_onlevel_images(self, source_dirname, destination_dirname, name_included):
		"""Prepara las imagenes para realizar predicciones sobre el modelo, la imagenes deben estar
		dentro de una carpeta
		Args:
			source_dirname: string; ruta en crudo de la carpeta contenedora de imagenes.
			destination_dirname: string; ruta en crudo donde serán guardadas las imagenes
			procesadas.
			name_included: boolean; especifica que el nombre clase de cada imagen se encuentra
			en el nombre de la misma (True) de lo contrario se especifica (False).
			"""
		print("Trabajando..")
		start = time.time()
		contador = 0
		positive_faces = 0
		try:
			for image in os.listdir(source_dirname):
				full_image_path = os.path.join(source_dirname, image)
				if not self.__verify_image(full_image_path):
					print("la imagen contiene errores y no será procesada {0}".format(image))
					continue

				image_class_name = self.__get_img_name(image) if name_included else image[:image.find(".")]
				if self.__faceDetector(full_image_path, image_class_name, destination_dirname, contador):
					self.writeOnFile(os.path.basename(image), image_class_name+"_"+str(contador)+".jpg", "imagenes_procesadas_faltantes_hombre_unir.txt")
					contador += 1
				elif self.__dlib_detector(full_image_path, image_class_name, destination_dirname, contador):
					self.writeOnFile(os.path.basename(image), image_class_name+"_"+str(contador)+".jpg", "imagenes_procesadas_faltantes_hombre_unir.txt")
					contador += 1
				else:
#					movemos la imagen que no tuvo algún rostro detectado
#					name_image = image[:image.find(".")]  obtiene solo el nombre de la imagen sin extensión
					self.__create_folder(self.__nofaces_dirname)
					shutil.move(full_image_path, self.__nofaces_dirname)
			print("{0} rostros procesados.".format(contador))
			end = time.time()
			print("----segundos {0} ----".format(end-start))
		except IOError:
			print("Error(resize_onlevel)-->{0}".format(sys.exc_info()))


	def __save_image(self, full_image, class_name, destination_dirname, contador):
		"""Gurada las imagenes de los rostros en blanco y negro con la dimension
		especificada
		Args:
			full_image: opencv_object; objeto opencv de la imagen
			class_name: string; nombre de la clase a la cual pertenece el rostro
			destination_dirname: string; ruta del directorio donde la imagen será guardada
			contador: Integer; número de imagen (se basa en el número de imagenes que
			contenga el directorio especificado en la variable DIR_TRAIN/mujer-hombre)
		"""
		#tamaño de las imágenes: height, width
		try:
			self.__create_folder(destination_dirname)
			#cambiamos el tamaño de la imagen
			resize = self.__resize_keep_aspect_ratio(full_image)
			cv.imwrite(os.path.join(destination_dirname, class_name + "_" + str(contador)+".jpg"), resize)
		except IOError:
			print("save_image: Error al intentar gaurdar la imagen: {0}".format(sys.exc_info()))


	@staticmethod
	def writeOnFile(basename_img, newbasename_img, log_txt_name):
		"""Crea un registro (txt) con el nombre anterior de la imagen y su nuevo nombre
		de la forma: newbasename_img*basename_img
		Args:
			basename_img: string; nombre original de la imagen
			newbasename_img: string; nuevo nombre de la imagen
			log_name: string; nombre del archivo a crear, e.g
			imagenes_procesadas_test, imagenes_procesadas_train"""

		assert os.path.splitext(log_txt_name)[1].lower() == ".txt", "el formato de archivo debe ser .txt"
		txt_file = os.path.join("/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/", log_txt_name)
		#comprobamos que el archivo sea como tal un archivo con isfile
		#si deseamos comprobar si existe podemos usar file.exists()-True-False
		#si el archivo existe agregamos información al final del archivo de lo contrario se crea el archivo
		if os.path.exists(txt_file):
			with open(txt_file, "a") as f:
				f.write(newbasename_img+"*"+basename_img+"\n")
		else:
			with open(txt_file, "w") as f:
				f.write(newbasename_img+"*"+basename_img+"\n")


	def __get_img_name(self, img_name):
		"""Asigna la clase hombre|mujer de acuerdo al nombre que contenga la imagen: imagen-mujer234.jpg
		-> return mujer
		Args:
			img_name: string; nombre de la imagen
		Return:
			hombre|mujer: string; clase correspondiente de acuerdo a nombre de la imagen."""
		if "mujer" in os.path.basename(img_name):
			return "mujer"
		elif "hombre" in os.path.basename(img_name):
			return "hombre"
#		levantar una excepción sino esta nombrada correctamente con mujer/hombre


	def get_img_size(self, source_dirname):
		"""Busca que las imagenes cumplan el tamaño de dimensiones mayor a 32x32 píxeles, las imagenes
		que no cumplan este requisito serán registradas en images_noaccepted_size.txt
		Args:
			source_dirname: string; ruta en crudo de la carpeta a examinar, esta debe estar con el
			formato:
			..directorio/mujer/imagen_x.jpg
			..directorio/hombre/imagen_x.jpg"""
		contador = 0
		for classes in os.listdir(source_dirname):
			if len(classes) > 0:
				clas = classes.strip('\\')
				source_dirname_gender = os.path.join(source_dirname, clas)
				self.writeOnFile(source_dirname, clas, "images_noaccepted_size.txt")
				for img in os.listdir(source_dirname_gender):
					full_img_path = os.path.join(source_dirname_gender, img)
					img_opencv = self.__get_correct_color_format(full_img_path)
					height, width = img_opencv.shape[:2]
					if height <= 32 and width <= 32:
#						self.writeOnFile(img, ".", "images_noaccepted_size.txt")
						contador += 1
			print(contador)

