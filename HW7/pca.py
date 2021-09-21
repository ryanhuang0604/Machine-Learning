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


class PCA:
	def __init__(self, k):
		self.k = k

	def mean(self, data):
		return np.mean(data, axis=1)

	def scatter_matrix(self, data):
		""" S = sum((Xk-m)@(Xk-m)^T) / n, where k=1,...,n """
		return np.cov(data, bias=True)

	def find_k_largest_eigenvalues(self, cov):
		print('Calculating eigen values and vectors...')
		k = self.k
		eigen_value, eigen_vector = np.linalg.eig(cov)
		sorting_index = np.argsort(-eigen_value)
		eigen_value = eigen_value[sorting_index]
		eigen_vector = eigen_vector.T[sorting_index]

		return eigen_value[0:k], (eigen_vector[0:k])

	def transform(self, W, data):
		""" return W @ data """
		return W.T @ W @ data

	def pca_main(self, data):
		""" data => (d,n) (45045, 135)"""
		
		'''
		### accelerate ###

		# mean
		mean = self.mean(data)	# (45045, 1)
		data = data.copy() - mean	# n*d

		# S(covariance)
		S = self.scatter_matrix(data.T)		#(45045, 45045) -> (135, 135)

		# eigenvector & eigenvalue -> principle components
		eigen_value, eigen_vector = self.find_k_largest_eigenvalues(S)	# (25, 135)
		eigen_vector = (data @ eigen_vector.T).T

		### end ###
		'''
		
		### original ###

		S = self.scatter_matrix(data)
		eigen_value, eigen_vector = self.find_k_largest_eigenvalues(S)	# (25, 135)

		### end ###

		print("eigen_value:")
		print(eigen_value)
		print("eigen_vector:")
		print(eigen_vector.shape)
		# Now W is eigen_vector (25, 45045)

		'''
		fig, axes = plt.subplots(5, 5)
		for i in range(25):
			axes[int(i / 5), i % 5].imshow(eigen_vector[i].reshape(195, 231), cmap="gray")
		plt.show()
		'''
		
		transformed_data = self.transform(eigen_vector, data)
		transformed_data = np.real(transformed_data)
		print(transformed_data)

		return transformed_data.T


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


def kernel_PCA(imgData_train, gamma,alpha, n_components, kernel_type):
	#imgData_mean = np.mean(imgData_train, axis=1).reshape(-1, 1)

	K = kernel_function(imgData_train, gamma, alpha, kernel_type)

	# Center the kernel matrix.
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)   

	# Obtaining eigenpairs from the centered kernel matrix
	# scipy.linalg.eigh returns them in ascending order
	eigenvalues, eigenvectors = eigh(K)
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

	pca_model = PCA(k)
	transformed_data = pca_model.pca_main(face_matrix.T)	# (135, 45045) -> (135, 45045)
	print("transformed data: ", transformed_data.shape)

	fig, axes = plt.subplots(2, 10)
	idx = np.random.choice(135, 10, replace=False)
	print(idx)
	for i, random_idx in enumerate((idx)):
		axes[0, i].imshow(face_matrix[random_idx].reshape(195, 231), cmap="gray")
		axes[1, i].imshow(transformed_data[random_idx].reshape(195, 231), cmap="gray")
	plt.show()