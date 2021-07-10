import os
import sys, traceback
import random

import cv2 as cv
from preprare_image import PrepareImage

class DataRead:

	def __init__(self):
		self._labels = {'hombre': 0, 'mujer': 1}
		self._inferencia_log_file = "inferencia_log.txt"
		self._root_path = os.path.dirname(os.path.realpath(__file__))


	def _get_class_label(self, class_name):
		return self._labels[class_name]


	def count_tfrecords_files(self, source_directory):
		"""
		counts how many tfrecord files are inside of a directory, the method assumes all
		files inside the directory are in such format.
		Args:
			source_directory: string, path of the directory containing the tfrecord 
			files.
		Return:
			int: number of files inside of the directory.
		"""
		number_of_files = 0
		for tf_file in os.listdir(os.path.dirname(source_directory)):
			if tf_file.lower().endswith('.tfrecord'):
				number_of_files += 1
		return number_of_files


	def read_byname(self, source_dirname):
		"""
		Gives to the network a pipeline for prediction on unseen images.
		NOTE: Gives to the network a pipeline for prediction on unseen images. Unlike
		read_file method, this method returns the labels for each image based in the
		image's name, the image name should be in the form: woman_0: mujer_0.jpg,
		hombre_0.jpg. Where '_' is the delimiter that separates the class of the image
		from the rest of the image name.
		"""
		absolute_path_images = []
		labels = []
		if os.path.exists(source_dirname):
			for image in os.listdir(source_dirname):
				labels.append(self._get_class_label(os.path.basename(image).split('_')[0]))
				absolute_path_images.append(os.path.join(source_dirname, image))

			return absolute_path_images, labels
		else:
			print("The directory does not exists or the path is wrong -->{0}".format(source_dirname))


	def read_bydirectories(self, source_dirname):
		"""	
		Gives to the network a pipeline for prediction on unseen images.
		Images and the directories should be in the following structure:
		directory-
			..directorio_clase_1/imagen_x.jpg
			..directorio_clase_2/imagen_x.jpg
		"""
		if os.path.exists(source_dirname):
			absolute_path_images = []
			labels = []
			for classes in os.listdir(source_dirname):
				if len(classes) > 0:
					class_name = classes.strip('\\')
					source_dirname_gender = os.path.join(source_dirname, class_name)
					for image in os.listdir(source_dirname_gender):
						absolute_path_images.append(os.path.join(source_dirname_gender, image))
						labels.append(self._get_class_label(class_name))

			return absolute_path_images, labels
		else:
			print("The directory does not exists or the path is wrong -->{0}".format(source_dirname))


	def read_byrandom(self, source_dirname):
		"""	
		Gives to the network a pipeline for prediction on unseen images.
		This method generates labels for each image with random values of: 0 (for men) 
		and 1 (for women), assuming a real case where is unknown the class of the 
		image to be inferred.	
		Args:
			source_dirname: string; directory path that contains the images to be 
			predicted.	
		Return:
			absolute_path_images: list; path for each image.
			labels: list; random generated labels for the images.
		"""
		absolute_path_images = []
		labels = []
		for image in os.listdir(source_dirname):
			absolute_path_images.append(os.path.join(source_dirname, image))
			labels.append(random.randrange(0, 2))
		return absolute_path_images, labels


	def save_image(self, full_image_path, predicted_label, image_index):
		"""
		Saves the image based on performed prediction in the directory of it's
		corresponding class: directory_data/mujer/..  | directory_data/hombre/..
		Args:
			full_image_path: string; path of the image.
			predicted_label: int; inferred class by the model 0|1.
			image_index: int; number of the image which was inferred.
		"""
		if os.path.exists(full_image_path):
			image_name = os.path.basename(full_image_path)
			cv_image = cv.imread(full_image_path, 0)
			inference_dir_path = os.path.join(self._root_path,"/inferencia/")

			if predicted_label == 1:
				inference_directory_fullpath = self._create_dirs(inference_dir_path, "mujer")
				new_image_name = "mujer_"+str(image_index)+".jpg"
				try:
					cv.imwrite(os.path.join(inference_directory_fullpath, new_image_name), cv_image)
				except IOError:
					print("An error ocurred while trying to save the image->{0}".format(sys.exc_info()))
				PrepareImage().writeOnFile(image_name, new_image_name, self._inferencia_log_file)
			elif predicted_label == 0:
				inference_directory_fullpath = self._create_dirs(inference_dir_path, "hombre")
				new_image_name = "hombre_"+str(image_index)+".jpg"
				try:
					cv.imwrite(os.path.join(inference_directory_fullpath, new_image_image), cv_image)
				except IOError:
					print("An errro ocurred while trying to save the image->{0}".format(sys.exc_info()))
				PrepareImage().writeOnFile(image_name, new_image_image, self._inferencia_log_file)
		else:
			print("The file does not exists or the path is wrong-->{0}".format(full_image_path))


	def _create_dirs(self, directory_path, gender):
		gender_directory_fullpath = os.path.join(directory_path, gender)
		if not os.path.exists(gender_directory_fullpath):
			os.makedirs(gender_directory_fullpath)

		return gender_directory_fullpath
