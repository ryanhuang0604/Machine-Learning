import numpy as np
from itertools import chain
import imageio
import os
import glob
import random
import matplotlib.pyplot as plt


def loadImgData():
	n = 135
	filenames = [img for img in glob.glob("Yale Face Database/Training/*.pgm")]
	m = [[] for i in range(n)]
	for i in range(n):
		m[i] = list(chain.from_iterable(imageio.imread(filenames[i])))

	m = np.matrix(m)	# (135, 45045)

	return m


class LDA:
	def __init__(self, k, label):
		self.k = k
		self.label_min = int(np.min(label))
		self.label_max = int(np.max(label))
		self.class_num = int(np.max(label)) - int(np.min(np.min(label))) + 1

	def class_mean(self, data, label):
		class_mean = []
		for i in range(self.label_min, self.label_max + 1):
			class_mean.append(np.mean(data[label == i], axis=0))

		return np.array(class_mean)

	def overall_mean(self, data):
		return np.mean(data, axis=0)

	def within_class_scatter(self, data, label):
		""" sum of each class scatter """

		d = data.shape[1]
		within_class_scatter = np.zeros((d, d))
		for i in range(self.label_min, self.label_max + 1):
			within_class_scatter += np.cov(data[label == i].T)

		return np.array(within_class_scatter)

	def in_between_class_scatter(self, data, label, class_mean, overall_mean):
		""" sum(nj * (mj-m)@(mj-m)^T), where j=class index """

		class_data_cnt = []
		for i in range(self.label_min, self.label_max + 1):
			class_data_cnt.append(list(label).count(i))
		class_data_cnt = np.array(class_data_cnt)

		d = data.shape[1]
		in_between_class_scatter = np.zeros((d, d))
		for i in range(self.class_num):
			print(i, ":")
			# print(class_mean[i])
			# print(overall_mean)
			class_mean_col = class_mean[i].reshape(d, 1)
			overall_mean_col = overall_mean.reshape(d, 1)
			tmp = (class_mean_col - overall_mean_col) @ (class_mean_col - overall_mean_col).T
			print(tmp.shape)
			print('-------------')
			in_between_class_scatter += class_data_cnt[i] * (class_mean_col - overall_mean_col) @
				(class_mean_col - overall_mean_col).T
		in_between_class_scatter = np.array(in_between_class_scatter)
		return in_between_class_scatter

	def find_k_largest_eigenvalues(self, cov):
		k = self.k
		eigen_value, eigen_vector = np.linalg.eig(cov)
		sorting_index = np.argsort(-eigen_value)
		eigen_value = eigen_value[sorting_index]
		eigen_vector = eigen_vector.T[sorting_index]

		return eigen_value[0:k], (eigen_vector[0:k])

	def transform(self, W, data):
		return W @ data

	def lda_main(self, data, label):
		# overall mean
		overall_mean = self.overall_mean(data)
		
		# class mean
		class_mean = self.class_mean(data, label) 

		# within-class scatter matrix
		within_class_s = self.within_class_scatter(data, label) 
		print("within_class:")
		print(within_class_s.shape)

		# in-between-class scatter matrix
		in_between_class_s = self.in_between_class_scatter(data, label, class_mean, overall_mean)
		print("in_between_class:")
		print(in_between_class_s.shape)

		# eigenvalues & eigenvectors -> first k largest
		eigen_value, eigen_vector = self.find_k_largest_eigenvalues(np.linalg.pinv(within_class_s) @ in_between_class_s)
		print("eigen_vector:")
		print(eigen_vector.shape)
		print(eigen_vector)

		transformed_data = self.transform(np.real(eigen_vector), data.T)
		print("transformed_data:")
		print(transformed_data)
		
		return transformed_data


def kernel_function(X,gamma,alpha,kernel_type):
	
	sq_dists = pdist(X, 'sqeuclidean')  
	print(sq_dists.shape)

	# Convert pairwise distances into a square matrix
	mat_sq_dists = squareform(sq_dists)
	print(mat_sq_dists.shape)

	# Compute the symmetric kernel matrix
	# Radial basis function
	if kernel_type == '1':
		K = np.exp(-gamma * mat_sq_dists)
	# Quadratic basis function
	elif kernel_type == '2':
		K = (1 + gamma*mat_sq_dists/alpha)**(-alpha)

	return K


def kernel_LDA(imgData_train, gamma,alpha, n_components, kernel_type):
	# in the MxN dimensional dataset.
	n = imgData_train.shape[0]
	M = np.zeros((n, n))
	N = np.zeros((n, n))
	mean_classes = np.zeros((n, 15))	# 15 classes

	Ms = kernel_function(imgData_train,gamma,alpha,kernel_type)
	# all classes mean
	for i in range(imgData_train.shape[1]):
		mean_classes[:, train_labels[i]] += Ms[:, train_labels[i]]
	mean_classes = mean_classes / 10
	
	imgData_mean = np.mean(Ms, axis=1).reshape(-1, 1)
  
	# distance between-class scatter
	for i in range(15):
		d = mean_classes[:, i].reshape(-1,1) - imgData_mean
		M += 10 * d @ d.T
	
	N = Ms.shape[0]
	one_n = np.ones((N,N)) / N
	I = np.identity(N)

	# distance within-class scatter
	for i in range(imgData_train.shape[1]):
		d = Ms[:, train_labels[i]].reshape(-1,1)*(I-one_n)*Ms[:, train_labels[i]].reshape(-1,1).T
		N += d 
 
	# Obtaining eigenpairs from the centered kernel matrix
	# scipy.linalg.eigh returns them in ascending order
	eigenvalues, eigenvectors = eigh(np.linalg.pinv(N)@M)
	eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1] 
	eigenvectors = np.column_stack([eigenvectors[:, i] for i in range(n_components)])    
	
	return eigenvectors


def testing_accuracy(imgData_test, test_labels, y_train, train_labels, W, imgData_mean=None, k=1):
	
	if imgData_mean is None:
		imgData_mean = np.zeros((imgData_test.shape[0],1))

	y_test = W.T @ (imgData_test - imgData_mean)

	# k-nn classifier
	predicted_labels = np.zeros(y_test.shape[1])
	for i in range(y_test.shape[1]):
		distance = np.zeros(y_train.shape[1])
		for j in range(y_train.shape[1]):
			distance[j] = np.sum(np.square(y_test[:,i] - y_train[:,j]))
		sort_index = np.argsort(distance)
		nearest_neighbors = train_labels[sort_index[:k]]
		unique, counts = np.unique(nearest_neighbors, return_counts=True)
		nearest_neighbors = [k for k,v in sorted(dict(zip(unique, counts)).items(), key=lambda item: -item[1])]
		predicted_labels[i] = nearest_neighbors[0]

	accuracy = np.count_nonzero((test_labels - predicted_labels) == 0) / len(test_labels)
	
	return accuracy


if __name__ == '__main__':
	k = 25

	face_matrix = loadImgData()		# (135, 45045)

	lda_model = LDA(k, label)
	transformed_data = lda_model.lda_main(face_matrix.T, label)
	print("transformed data: ", transformed_data.shape)

	fig, axes = plt.subplots(2, 10)
	idx = np.random.choice(135, 10, replace=False)
	print(idx)
	for i, random_idx in enumerate((idx)):
		axes[0, i].imshow(face_matrix[random_idx].reshape(195, 231), cmap="gray")
		axes[1, i].imshow(transformed_data[random_idx].reshape(195, 231), cmap="gray")
	plt.show()