import os.path
import struct as st
import numpy as np
import math
import pandas
import matplotlib.pyplot as plt


def ImgDataGen(filepath):
	train_imagesfile = open(filepath, 'rb')

	# Read the magic number
	train_imagesfile.seek(0)
	magic = st.unpack('>4B', train_imagesfile.read(4))

	# Read the dimensions of the Image data-set
	nImg = st.unpack('>I', train_imagesfile.read(4))[0]  #num of images
	nR = st.unpack('>I', train_imagesfile.read(4))[0]  #num of rows
	nC = st.unpack('>I', train_imagesfile.read(4))[0]  #num of column

	# Reading the Image data
	nBytesTotal = nImg * nR * nC * 1  #since each pixel data is 1 byte
	images_array = np.asarray(st.unpack('>' + 'B' * nBytesTotal, train_imagesfile.read(nBytesTotal))).reshape((nImg, nR, nC))

	# Tally into 2 bins
	images_array = np.floor(images_array / 128)

	# flattering images_array: 60000 * 28 * 28 => 60000 * 784
	# [3, 6] => (3-1) * 28 + 6
	images_array = images_array.reshape(images_array.shape[0], images_array.shape[1] * images_array.shape[2])

	train_imagesfile.close()


	return images_array


def LabelDataGen(filepath):
	train_labelfile = open(filepath, 'rb')

	# Read the magic number
	train_labelfile.seek(0)
	magic = st.unpack('>4B', train_labelfile.read(4))

	# Read the number of data-set items
	nItem = st.unpack('>I', train_labelfile.read(4))[0]  #num of items

	# Reading the Label data
	nBytesTotal = nItem * 1  #since each item is 1 byte
	labels_array = np.asarray(st.unpack('>' + 'B' * nBytesTotal, train_labelfile.read(nBytesTotal))).reshape((nItem))

	train_labelfile.close()


	return labels_array


def readData():
	train_file = {'images': 'train-images-idx3-ubyte', 'labels': 'train-labels-idx1-ubyte'}
	test_file = {'images': 't10k-images-idx3-ubyte', 'labels': 't10k-labels-idx1-ubyte'}

	if os.path.exists('train_image.npy'):
		train_image_array = np.load('train_image.npy')
	else:
		train_image_array = ImgDataGen(train_file['images'])
		np.save('train_image', train_image_array)

	if os.path.exists('train_label.npy'):
		train_label_array = np.load('train_label.npy')
	else:
		train_label_array = LabelDataGen(train_file['labels'])
		np.save('train_label', train_label_array)

	# Read data from testing files
	if os.path.exists('test_image.npy'):
		test_image_array = np.load('test_image.npy')
	else:
		test_image_array = ImgDataGen(test_file['images'])
		np.save('test_image', test_image_array)

	if os.path.exists('test_label.npy'):
		test_label_array = np.load('test_label.npy')
	else:
		test_label_array = LabelDataGen(test_file['labels'])
		np.save('test_label', test_label_array)


	return train_image_array, train_label_array, test_image_array, test_label_array


class BernoulliEM():
	def __init__(self, train_data):
		self.train_data = train_data
		self.image_cnt = train_data.shape[0]  # 60000
		self.pixel_cnt = train_data.shape[1]  # 784
		self.lambd = np.full((10), 0.1)  # {0, 1, 2, 3, 4, ..., 9} -> lambda: [0.1, 0.1, ..., 0.1]
		self.P = np.random.rand(10, self.pixel_cnt)  # (each pixel | #) -> 10*784


	def E_step(self):
		image_cnt = self.image_cnt
		pixel_cnt = self.pixel_cnt
		'''
		Expectation: calculate responsibility
		w(x1 | #0), w(x1 | #1), ..., w(x1 | #9)
		.
		.
		.
		w(x60000 | #0), w(x60000 | #1), ..., w(x60000 | #9)
		wi = exp(Zi) / sum(exp(Zi)), where Zi is log of probability
		'''
		z = np.zeros((image_cnt, 10))  # z -> 60000*10
		w = np.zeros((image_cnt, 10))  # w -> 60000*10
		for i in range(image_cnt):  # 60000
			data = self.train_data[i]  # data -> 784 pixels
			for j in range(10):
				z[i][j] = np.log(self.lambd[j]) + np.sum(self.P[j] * data) + np.sum((1 - self.P[j]) * (1 - data))
			w[i] = np.exp(z[i] - max(z[i]))
			w[i] /= np.sum(w[i])

		return w


	def M_step(self, w):	# 找出在這組w下 P,λ的 weight likelihood
		image_cnt = self.image_cnt
		pixel_cnt = self.pixel_cnt
		'''
		Maximization: update lambd, P
		lambd -> 10
		P -> 10*784
		w -> 60000*10
		train_data -> 60000*784
		'''
		self.lambd = np.sum(w, axis=0) / image_cnt	# lambda = Σw / N
		self.P = (w.T @ self.train_data)	# P = Σwixi / Σwi
		w_sum = np.sum(w, axis=0)
		for i in range(10):
			self.P[i] /= w_sum[i]


	def EM_train(self):
		last_lambd = self.lambd.copy()
		last_P = self.P.copy()
		total_iterate = 0
		while (True):
			total_iterate += 1
			w = self.E_step()
			self.M_step(w)
			if np.linalg.norm(self.lambd - last_lambd) <= 0.1 and np.linalg.norm(self.P - last_P) <= 0.1:
				break
			last_lambd = self.lambd.copy()
			last_P = self.P.copy()


		return self.lambd, self.P, total_iterate


def predict(data, lambd, P):
	'''
	data -> single image, 784 pixels
	lambd -> 10
	P -> 10 * 784
	'''
	predict_P = np.zeros((10))
	for i in range(10):
		predict_P[i] = lambd[i] * np.prod(P[i][data == 1])

	predict_class = np.argmax(predict_P)


	return predict_class


def label_class_map(image, label, lambd, P):
	'''
	generate label-class mapping, Ex: label[0] = class 5
	'''
	label_class_map = np.full((10), -1)

	appear_of_predict = np.zeros((10))
	for i in range(10):
		label_i_data = image[label == i]
		for j in label_i_data:
			predict_class = predict(j, lambd, P)
			appear_of_predict[predict_class] += 1

		# choose label_i's class
		sort_predict = np.argsort(appear_of_predict)

		idx = 9
		while np.any(label_class_map == sort_predict[idx]):
			idx -= 1
		label_class_map[i] = sort_predict[idx]
		appear_of_predict = np.zeros((10))


	return label_class_map


def printPredictNumber(P, mapping):
	for i in range(10):
		class_i = mapping[i]
		display_pixels = P[class_i].copy()
		display_pixels[P[class_i].copy() > 0.5] = 1
		display_pixels[P[class_i].copy() <= 0.5] = 0
		display_pixels = display_pixels.reshape(28, 28)
		print("class {}:".format(i))

		display = pandas.DataFrame(display_pixels, dtype=int)
		print(display.to_string(index=False))
		print("\n")


def confusionMatrix(image, label, lambd, P, mapping):
	'''
					Predict number i Predict not number i
	Is number i          ...               ...
	Isn't number i       ...               ...
	'''
	for i in range(10):
		print("\n------------------------------------------------------------\n")
		print("Confusion Matrix {}:".format(i))
		confusion_matrix = np.zeros((2, 2))

		is_number_i = image[label == i]
		isnot_number_i = image[label != i]
		for data in is_number_i:
			predict_class = predict(data, lambd, P)
			predict_number = np.argwhere(mapping == predict_class).item()
			if (predict_number == i):
				confusion_matrix[0][0] += 1
			else:
				confusion_matrix[0][1] += 1
		for data in isnot_number_i:
			predict_class = predict(data, lambd, P)
			predict_number = np.argwhere(mapping == predict_class).item()
			if (predict_number == i):
				confusion_matrix[1][0] += 1
			else:
				confusion_matrix[1][1] += 1

		confusion_matrix_result = pandas.DataFrame(confusion_matrix, columns=['Predict cluster {}'.format(i), 'Predict cluster {}'.format(i)], index=['Is number {}'.format(i), "Isn't number {}".format(i)])
		print("\n")
		print(confusion_matrix_result)
		print("\nSensitivity (Successfully predict number {}):".format(i), confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
		print("Specificity (Successfully predict not number {}):".format(i), confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))


if __name__ == '__main__':
	# Open the IDX file in readable binary mode.
	train_image_array, train_label_array, test_image_array, test_label_array = readData()

	partial_image = train_image_array
	partial_label = train_label_array

	# Training
	EM_model = BernoulliEM(partial_image)
	lambd, P, total_iterate = EM_model.EM_train()

	# generate label-class-mapping
	mapping = label_class_map(partial_image, partial_label, lambd, P)

	# print predict number and confusion matrix
	printPredictNumber(P, mapping)
	confusionMatrix(partial_image, partial_label, lambd, P, mapping)

	# predict testing data
	correct = 0
	for i in range(500):
		predict_class = predict(test_image_array[i], lambd, P)
		predict_number = np.argwhere(mapping == predict_class).item()
		label_number = test_label_array[i]

		if predict_number == label_number:
			correct += 1

	print("\n")
	print("Total Iteration to converge:", total_iterate)
	accuracy = correct / 500
	print("Total error rate:", 1 - accuracy)