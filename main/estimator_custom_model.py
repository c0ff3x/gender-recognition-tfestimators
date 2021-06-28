"""*****Código de clasificación de género principal para entrenar la red con tamaño de imagen 128*****
-Realizar prediccines sobre img_list para obtener más imagenes de cada clase y agregarlas a los archivos
de entrenamiento y comparar con myfourth_model si la presición del modelo mejora aun más.(los datos están redimensionados
antes de ser procesados.)
-Agregar más imagenes para entrenamiento porque no realiza bien las predicciones además se debe buscar
que significa cuando la presición de predicción es menor que la prediccion de test.
-Porque son diferentes los resultados de training, validation and test
-Salvar el modelo de tf.estimator
-Revisar en la documentación oficial de tensorflow el como guardar un modelo para
usarlo en otro lenguaje (deploy)
-Tenemos que revisar lo que tiene google colaboratory y este código para ponerlos en sincronía y ejecutar el modelo!
-Buscar técnicas que ayuden a mejorar la precisión del modelo como batch_normalization.
-Qué es confusion matrix, ¿en que puede ayudar?
-what is a roc cruve on machine learning
**Listo
-Estamos enfrentando el detectar rostros que no están rectos a la cámara y tiene inclinaciones ligeras  entre 1 y 90 grados
debemos de poder detectar dichos rostros además de ver como mejorar la predicción del modelo procesando la imagen con
el preprocesamiento de imágenes del pdf "Image Enhancement for Face Recognition"(optional).--> data
augment obtener más cantidad de datos al realizar esta operación sobre el dataset exsitente.
"""
#tensorflow version used = 1.13.1
import tensorflow as tf
import os
import sys, traceback
import numpy as np
from data_read import *


class Estimator_Gender_model:
	def __init__(self):
		self.__train_tf_files = '/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/train_data/*.tfrecord'
		self.__test_tf_files = "/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/test_data/*.tfrecord"
		self.__BATCH_SIZE = 10
		self.__IMAGE_SIZE = 128
		self.__EPOCHS = 6600
		self.__to_train = True
		self.__output_dir = "/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/new_mygender_chkpoint_model"
		self.__LEARNING_RATE = 1e-4
		self.__drop1 = 0.25
		self.__drop2 = 0.4
		self.__ground_label = []
		self.__images = []


	def input_fn(self):
		""" Crea la estructura de canal de entrada de datos para la red unsando la API tf.data
		Retorna:
		iterador: iterator, iterador que permite obtener el siguiente elemento dentro del dataset
		"""
		if self.__to_train:
#			dataset = tf.data.TFRecordDataset([self.tfrecord_file])
#			varajea los diferentes fragmentos tfrecord
			num_shards = count_files(self.__train_tf_files)
			dataset = tf.data.Dataset.list_files(self.__train_tf_files).shuffle(num_shards)
			dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=num_shards)
			dataset = dataset.shuffle(buffer_size=8125, reshuffle_each_iteration=True)
#			más grande que el tamaño de número de registros(imagenes) en un sólo fragmento para que mezcle los datos
#			https://github.com/tensorflow/tensorflow/blob/041ae08f82fb6cdda3236a890f8c46d3a75a8d3f/tensorflow/python/data/ops/dataset_ops.py#L1044
		else:
			num_shards = count_files(self.__test_tf_files)
			dataset = tf.data.Dataset.list_files(self.__test_tf_files).shuffle(num_shards)
			dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=num_shards)  #([self.tfrecord_file_eval])
			dataset = dataset.shuffle(buffer_size=4250, reshuffle_each_iteration=True)
		def decode_fn(tfrecord_file):
			""" Decodifica los elementos necesarios del archivo tfrecord para alimentar a la red
			Args:
				tfrecord_file: string, ruta de ubicación del archivo tfrecord --> /path/example/file.tfrecord
			Retorna:
				image: imagen jpeg
				label: int, identificador de la clase de la imagen.
			"""
			features = {
				'filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
				'format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
				'height': tf.FixedLenFeature([], tf.int64),
				'width': tf.FixedLenFeature([], tf.int64),
				'channels': tf.FixedLenFeature([], tf.int64),
				'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
				'label': tf.FixedLenFeature([], tf.int64)
			}
			sample = tf.parse_single_example(tfrecord_file, features)
#			image = tf.image.decode_image(sample['image/encoded'])
			image = sample['image']

			with tf.name_scope('image_decode_jpeg', [ image ], None):
				image = tf.image.decode_jpeg( image, channels = 1 )
				image = tf.image.convert_image_dtype( image, dtype = tf.float32 )

			img_shape = tf.stack([sample['height'], sample['width'], sample['channels']])
			image = tf.reshape(image, img_shape)
			label = tf.cast(sample['label'], tf.int64)

			return {"image":image}, label

		dataset = dataset.repeat(self.__EPOCHS)
		dataset = dataset.map(decode_fn, num_parallel_calls=2)
		dataset =  dataset.batch(self.__BATCH_SIZE)
#		`prefetch` permite que el conjunto de datos obtenga lotes, en segundo plano mientras el modelo está en formación.
		dataset = dataset.prefetch(1)
#		iterator = dataset.make_one_shot_iterator()
		return dataset


#	recontrucción de la red convolucional
	def cnn_model_fn(self, features, labels, mode):
		input_layer = tf.reshape(features["image"], [-1, self.__IMAGE_SIZE, self.__IMAGE_SIZE, 1])
		with tf.name_scope("ConvPoolLayer1"):
			conv1 = tf.layers.conv2d(inputs = input_layer, filters = 32, kernel_size = [3,3], strides=1, padding = "SAME", activation = tf.nn.relu)

#			conv2 = tf.layers.conv2d(inputs = conv1, filters = 32, kernel_size = [3,3], strides=1, padding = "SAME", activation = tf.nn.relu)

			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, padding = "SAME")
#			dropout1 = tf.layers.dropout(pool1, self.__drop1, training=mode == tf.estimator.ModeKeys.TRAIN)
		with tf.name_scope("ConvPoolLayer2"):
			conv3 = tf.layers.conv2d(inputs = pool1, filters = 64, kernel_size=[3,3], strides=1, padding="SAME",activation=tf.nn.relu)

#			conv4 = tf.layers.conv2d(inputs = conv3, filters = 64, kernel_size=[3,3], strides=1, padding="SAME",activation=tf.nn.relu)

			pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2, padding = 'SAME')
#			dropout2 = tf.layers.dropout(pool2, self.__drop1, training=mode == tf.estimator.ModeKeys.TRAIN)
		with tf.name_scope("ConvPoolLayer3"):
			conv5 = tf.layers.conv2d(inputs = pool2, filters = 128, kernel_size = [3,3], strides=1, padding = "SAME", activation = tf.nn.relu)

#			conv6 = tf.layers.conv2d(inputs = conv5, filters = 128, kernel_size = [3,3], strides=1, padding = "SAME", activation = tf.nn.relu)

			pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2,2], strides=2, padding='SAME')
			#dropout3 = tf.layers.dropout(pool3, self.__drop1, training=mode == tf.estimator.ModeKeys.TRAIN)
		with tf.name_scope("ConvPoolLayer4"):
			conv7 = tf.layers.conv2d(inputs = pool3, filters = 256, kernel_size = [3,3], strides=1, padding="SAME", activation=tf.nn.relu)

			pool4 = tf.layers.max_pooling2d(conv7, pool_size = [2, 2], strides = 2, padding="SAME")
#			#dropout4 = tf.layers.dropout(pool4, self.__drop1, training=mode == tf.estimator.ModeKeys.TRAIN)

		with tf.name_scope("ConvPoolLayer5"):
			conv8 = tf.layers.conv2d(inputs = pool4, filters = 512, kernel_size = [3,3], strides=1, padding="SAME", activation=tf.nn.relu)

			pool5 = tf.layers.max_pooling2d(conv8, pool_size = [2, 2], strides = 2, padding="SAME")

		with tf.name_scope("Flatten"):
#			shape = conv8.get_shape().as_list()
#			flat = tf.reshape(conv8, [-1, 8 * 8 * 512])
			flat = tf.layers.flatten(pool5)

		with tf.name_scope("DenseLayer"):
			full1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
			if self.__to_train:
				full_dropout = tf.layers.dropout(inputs=full1, rate=self.__drop2, training=mode == tf.estimator.ModeKeys.TRAIN)
			else:
				full_dropout = full1
			full2 = tf.layers.dense(inputs=full_dropout, units=1024, activation=tf.nn.relu)
#			full_dropout2 = tf.layers.dropout(inputs=full2, rate=self.__drop2, training=mode == tf.estimator.ModeKeys.TRAIN)

		with tf.name_scope("logits"):
			logits = tf.layers.dense(inputs=full2, units=2)

		try:
			global_step = tf.train.get_global_step()
			predicted_logit = tf.argmax(input=logits, axis=1,output_type=tf.int32)
			probabilities = tf.nn.softmax(logits)

			#predicciones
			predictions = {"predicted_logits" : predicted_logit,"probabilities" : probabilities}

			if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

			with tf.name_scope('loss'):
				cross_entropy = tf.losses.sparse_softmax_cross_entropy(
					labels=labels, logits=logits, scope='loss')
				tf.summary.scalar('loss', cross_entropy)

			with tf.name_scope('accuracy'):
				accuracy = tf.metrics.accuracy(
					labels=labels, predictions=predicted_logit, name='acc')
				tf.summary.scalar('accuracy', accuracy[1])


			#evaluación
			if mode == tf.estimator.ModeKeys.EVAL:
				return tf.estimator.EstimatorSpec(
					mode=mode,
					loss=cross_entropy,
					eval_metric_ops={'accuracy/accuracy':accuracy},
					evaluation_hooks=None)

			#creamos el optimizador
			optimizer = tf.train.AdamOptimizer(self.__LEARNING_RATE)
			train_op = optimizer.minimize(cross_entropy,global_step=global_step)

			#crea un hook para mostrar acc, loss & global step cada 100 iteraciones
			train_hook_list = []
			train_tensors_log = {'accuracy':accuracy[1],
					'loss':cross_entropy,
					'global_step':global_step}
			train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=100))

			if mode == tf.estimator.ModeKeys.TRAIN:
				return tf.estimator.EstimatorSpec(
					mode=mode,
					loss=cross_entropy,
					train_op=train_op,
					training_hooks=train_hook_list)
		except Exception as erro:
			print("Error-->{0}".format(sys.exc_info()))


	def read_file(self):
		"""
		VOY A ALIMENTAR A LA RED CON LAS IMAGENES PARA GENERAR PREDICCINOES PERO ANTES NECESITO
		SABER COMO GENERAR LOS "labels" VERDADEROS PARA CADA IMAGEN(me ayudo de attemp_create_tf..).
		use append for one element
		use extend for all elements >>>>imagenes_prueba_2 -data_set_positives -imagenes_listas
		Provee a la red un método de entrada para la prediccion en imagenes no vistas
		"""
		try:
			source_dirname = "/home/hambo-abadeer/Documentos/Bayot/PO/faceRecognition/imgalignceleb_processed/"
#			lista con la ruta de las imágenes
#			lista con la cantidad total de etiquetas de cada clase (0|1)
			relative_path_images, labels = read_byrandom(source_dirname)
			self.__images = relative_path_images
			self.__ground_label = labels

			path_imgs = tf.data.Dataset.from_tensor_slices((relative_path_images, labels))
			batch_size = 1
			imgs = path_imgs.map(self.load_and_preprocess_image)
			imgs = imgs.batch(batch_size).prefetch(None)
			return imgs
		except IOError:
			print("read_files:Error-->{0}".format(sys.exc_info()))


	def preprocess_image(self, image, label):
		image = tf.io.decode_jpeg(image, channels = 1)
		image = tf.image.resize_images(image, [self.__IMAGE_SIZE, self.__IMAGE_SIZE])
		return {"image":image}, label


	def load_and_preprocess_image(self, path, labels):
		image = tf.read_file(path)
		return self.preprocess_image(image, labels)


	def serving_input_fn(self):
		"""
		features = {
			'image': tf.FixedLenFeature([], dtype=tf.string, default_value='')
		}

		serialized_tf_example = tf.placeholder
		#Estoy terminando este método, especificamente si features debe tener los mismos parametros que
		#features del modelo en input_fn (primero realizar documentación, esto puede llevar tiempo)
		"""
		pass


	def main(self):
		try:
			classifier = tf.estimator.Estimator(
				model_fn=self.cnn_model_fn, model_dir=self.__output_dir)
			classifier.train(input_fn=lambda:self.input_fn(), steps=self.__EPOCHS)
			self.__to_train = False
			evaluation = classifier.evaluate(input_fn=lambda:self.input_fn(), steps=16)
			print(evaluation)
			"""

			self.__to_train = False
			predicted_classes = classifier.predict(input_fn=lambda:self.read_file())
			labels_predicted = []
#			la salida esta dada en  [hombre, mujer] donde 1 significa que dicha clase tuvo la probabilidad más alta
#			[1, 0] --> hombre, [0, 1] --> mujer
			for idx, prediction in enumerate(predicted_classes):
				labels_predicted.append(prediction["predicted_logits"])
				save_image(self.__images[idx], prediction["predicted_logits"], idx)

#			print("Ground labels {0} \nValores inferidos {1}".format(self.__ground_label, labels_predicted))
			print("Modelo usado {0}".format(self.__output_dir))
			assert len(self.__ground_label) == len(labels_predicted)
			aciertos = 0

			for i in range(len(self.__ground_label)):
				if self.__ground_label[i] == labels_predicted[i]:
					aciertos += 1
			precision = (aciertos/len(self.__ground_label)) * 100
			print("La precisión de la predicción es de",format(precision, "5.2f"))
			"""
		except tf.errors.OutOfRangeError:
			pass
		except IOError:
			print("Error-->{0}".format(sys.exc_info()))


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	Estimator_Gender_model().main()
#tf.app.run(main)
