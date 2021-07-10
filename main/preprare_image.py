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
		Initilize all creation paths for the generated directories and log files.
		destination_dirname: string; path where train/test images will be saved after
		pre-processing them.
		path_error_images: string; path where all the images with errors will be saved.
		(these images are not considered in train/test images.)
		nofaces_dirname: string; path where all images with no faces detected by opencv
		or dlib will be saved.
		"""
		self.__TRAIN_DIRECTORY = '../faceRecognition/data/'	
		self.__TEST_DIRECTORY = '../faceRecognition/testset/'
		self.__destination_dirname = os.path.dirname(os.path.realpath(__file__))+'/train/'
		self.__path_error_images= os.path.dirname(os.path.realpath(__file__))+'/Error_Images/'
		self.__nofaces_dirname = os.path.dirname(os.path.realpath(__file__))+"/train_negatives/"
		self.__face_cascade_path = cv.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
		self.__IMG_SIZE = 128
		self.__allowed_image_extention = ("jpg", "jpeg", "png")
		self.__full_image = []


	def __create_directory(self, directory_path):
		if not os.path.exists(directory):
			os.makedirs(directory)
		else:
			print("The directory already exists.")


	def __verify_image_has_no_error(self, image_path):
		"""Verifies that one image has no error and checks if it's format is one of the
		allowed formats.	
		Args:
			img_file: string; image's path to be analized."""
		try:
			sky_image = io.imread(image_path)
		except IOError:
			print("Couldn't read image: stack {0}".format(sys.exc_info()))
		if os.path.basename(image_path).lower().endswith(self.__allowed_image_extention):
			#width, height
			image_width, image_height = sky_image.shape[:2]
			if image_width > 20 and image_height > 20:
				return True
		else:
			print("Image should has at least one allowed image format, see docstring with python -i file.py")


	def __is_gray_scale(self, image_path):
		"""Verifies if the image is graycale or RGB traversing each pixel in the image and
		compares each one at each level if is equals to RVerifies if the image is graycalei
		or RGB traversing each pixel in the image and compares each one at each level if is
		equals to RGB..	
		Args:
			image: string; ruta Image to verify."""
		try:
			image = Image.open(image_path).convert('RGB')
			image_width, image_height = image.size
			for i in range(image_width):
				for j in range(image_height):
					r, g, b = image.getpixel((i, j))
					if r != g != b: return False
			return True
		except:
			print("is_gray_scale: Error-->{0}".format(sys.exc_info()))


	def __get_correct_color_format(self, image):
		"""Given an image path, opecv object loads each image with the 
		correct color format.
		Args:
			image: string, path of the image,
		Return:
			opencv_object, correct color format image (RGB or grayscale)."""
		if self.__is_gray_scale(image):
			return cv.imread(image, 0)
		else:
#			cv_image = self.__histo_enhancement(cv_image)
			return cv.imread(image, 1)


	def __faceDetector(self, raw_image, class_name, destination_dirname, contador):
		"""Detects faces inside an image using facecascade detector.
		Args:
			image: string; path of the image.
			classname: string, hombre|mujer:
			destination_dirname: string, directory path were the images will be saved.
			contador: integer, number of the image -> hombre_contador.jpg"""
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
		"""detects faces inside an image using dlib face detector.
		It's used like a second mechanism for face detectin if opencv face
		detector fails to detect a face inside an image.
		Args:
			image: string; path of the image.
			class_name: string; hombre|mujer.
			destination_dirname: string, directory path were the images will be saved.	
			contador: int; counter that will be part of the name of the image.
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
		"""Resizes an image keepig its aspect ratio.
		Args:
			image: opencv_object, image to be resized.
		Return:
			new_im: opencv_object, image with the new dimentions.
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
		"""Perform histogram equalization on the given an image.
		Args:
			img: opencv_object"""
		R, G, B = cv.split(img)
		output_R = cv.equalizeHist(R)
		output_G = cv.equalizeHist(G)
		output_B = cv.equalizeHist(B)
		output = cv.merge((output_R, output_G, output_B))
		return output


	def resize_images(self, source_dirname=None, destination_dirname=None):
		"""Given a directory in the form:
			../data/hombre/img-x.jpg
			../data/mujer/img-x.jpg
		it process the images for the train-tfrecord and test-tfrecord files,
		cutting the face, resizing it and chaning it's channel to 1
		Generates the directories:
			train/mujer/mujer-x.jpg
			train/hombre/hombre-x.jpg
			test/mujer/mujer-x.jpg
			test/hombre/hombre-x.jpg
		Args:
			source_dirname: string, directory containing the images to be
			processed a procesar.
			destination_dirname: string, directory where all the processed images
			will be saved.
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
		"""Prepares the images to be inferred with the model, the images must be inside a
		directory.	
		Args:
			source_dirname: string;  directory path that constains all the images.
			destination_dirname: string; directory path where all the processed images
			will be saved.
			name_included: boolean; specifies that the class name of each image is
			included in its name (True) otherwise (False).	
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
		"""Saves the face images in grayscale.
		
		Args:
			full_image: opencv_object; image's opencv object
			class_name: string; class name which the face image belongs.
			destination_dirname: string; directory path where the image will be
			saved.
			contador: Integer; image numberi it's based on the number of images
			that the directory ..TRAIN|TEST/classname has.	
		"""
		try:
			self.__create_folder(destination_dirname)
			#cambiamos el tamaño de la imagen
			resize = self.__resize_keep_aspect_ratio(full_image)
			cv.imwrite(os.path.join(destination_dirname, class_name + "_" + str(contador)+".jpg"), resize)
		except IOError:
			print("save_image: Error al intentar gaurdar la imagen: {0}".format(sys.exc_info()))


	@staticmethod
	def writeOnFile(basename_img, newbasename_img, log_txt_name):
		"""Creates a log file with the images previous name and the its new name in
		the form: newbasename_img*basename_img
		Args:
			basename_img: string; original image name
			newbasename_img: string; new name of the image
			log_name: string; name of the log file to create. e.g
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
		"""Assigns the classname (hombre|mujer) according to the name that the image has:
		imagen-mujer234.jpg -> return mujer
		Args:
			img_name: string; image name.
		Return:
			hombre|mujer: string; corresponding class according to the name of the image."""
		if "mujer" in os.path.basename(img_name):
			return "mujer"
		elif "hombre" in os.path.basename(img_name):
			return "hombre"
#		raise and exception if the image is not correctly named with mujer/hombre


	def get_img_size(self, source_dirname):
		"""Check that images fulfill the measurements greater than 32*32 pixels, the
		images that do not fulfill this requirement will logged in
		images_noaccepted_size.txt.
		Args:
			source_dirname: string; directory path to inspect, this must be with the
			following format:			:
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

