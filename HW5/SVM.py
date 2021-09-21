from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np
from scipy.spatial.distance import pdist


def gen_data(filename):
	return np.genfromtxt(filename, delimiter=',')


def train_with_default_kernels():
	y = gen_data('Y_train.csv')
	x = gen_data('X_train.csv')
	yt = gen_data('Y_test.csv')
	xt = gen_data('X_test.csv')
	compare = []

	"""
	-t 0: Default linear kernel
	-t 1: Default polynomial kernel
	-t 2: Default RBF kernel
	"""
	for i in range(3):
		param = '-t ' + str(i)
	
		model = svm_train(y, x, param)
		p_label, p_acc, p_val = svm_predict(yt, xt, model)
		compare.append(p_acc[0])
	
	print("Compare: ")
	print(compare)


def train_with_linear_kernel():
	y = gen_data('Y_train.csv')
	x = gen_data('X_train.csv')
	yt = gen_data('Y_test.csv')
	xt = gen_data('X_test.csv')
	"""
	Linear kernel: -t 0
	Applying grid search: c = {2^-5, 2^-3, 2^-1, ..., 2^11, 2^13, 2^15}
	"""
	C_grid_search = [2**c for c in range(-5, 16, 2)]
	compare = []
	for c_try in C_grid_search:
		param = '-t 0 -c {}'.format(c_try)
		model = svm_train(y, x, param)

		print("test: c = {}".format(c_try))
		p_label, p_acc, p_val = svm_predict(yt, xt, model)
		print(p_acc[0])
		compare.append(p_acc[0])
	print("\nCompare: ")
	print(compare)


def train_with_polynomial_kernel():
	y = gen_data('Y_train.csv')
	x = gen_data('X_train.csv')
	yt = gen_data('Y_test.csv')
	xt = gen_data('X_test.csv')
	"""
	Polynomial kernel: -t 1
	Applying grid search on c: c = {2^-5, 2^-3, 2^-1, ..., 2^11, 2^13, 2^15}
	Applying grid search on gamma: gamma = {2^-15, 2^-13, ..., 2^1, 2^3}
	"""
	C_grid_search = [2**c for c in range(-5, 16, 2)]
	Gamma_grid_search = [2**g for g in range(-15, 4, 2)]
	compare = []
	for c_try in C_grid_search:
		gamma_iter_compare = []
		for g_try in Gamma_grid_search:
			param = '-t 1 -c {} -g {}'.format(c_try, g_try)
			model = svm_train(y, x, param)

			print("test: c = {}, g = {}".format(c_try, g_try))
			p_label, p_acc, p_val = svm_predict(yt, xt, model)
			print(p_acc[0])
			gamma_iter_compare.append(p_acc[0])
		compare.append(gamma_iter_compare)
	print("\nCompare: ")
	print(compare)


def train_with_RBF_kernel():
	y = gen_data('Y_train.csv')
	x = gen_data('X_train.csv')
	yt = gen_data('Y_test.csv')
	xt = gen_data('X_test.csv')
	"""
	RBF kernel: -t 2
	Applying grid search on c: c = {2^-5, 2^-3, 2^-1, ..., 2^11, 2^13, 2^15}
	Applying grid search on gamma: gamma = {2^-15, 2^-13, ..., 2^1, 2^3}
	"""
	C_grid_search = [2**c for c in range(-5, 16, 2)]
	Gamma_grid_search = [2**g for g in range(-15, 4, 2)]
	compare = []
	for c_try in C_grid_search:
		gamma_iter_compare = []
		for g_try in Gamma_grid_search:
			param = '-t 2 -c {} -g {}'.format(c_try, g_try)
			model = svm_train(y, x, param)

			print("test: c = {}, g = {}".format(c_try, g_try))
			p_label, p_acc, p_val = svm_predict(yt, xt, model)
			print(p_acc[0])
			gamma_iter_compare.append(p_acc[0])
		compare.append(gamma_iter_compare)
	print("\nCompare: ")
	print(compare)


def train_with_precomputed_kernel():
	def linear_kernel(x1, x2):
		n1 = x1.shape[0]
		n2 = x2.shape[0]
		K_train = np.zeros((n1, n2 + 1))
		K_train[:, 1:] = np.dot(x1, x2.T)
		K_train[:, 0] = np.arange(n1) + 1
		return K_train

	def RBF_kernel(x1, x2, gamma):
		n1 = x1.shape[0]
		n2 = x2.shape[0]
		K_train = np.zeros((n1, n2 + 1))
		x1_norm = np.sum(x1**2, axis=-1)
		x2_norm = np.sum(x2**2, axis=-1)
		dist = x1_norm[:, None] + x2_norm[None, :] - 2 * np.dot(x1, x2.T)
		K_train[:, 1:] = np.exp(-gamma * dist)
		K_train[:, 0] = np.arange(n1) + 1
		return K_train

	def precomputed_kernel(x1, x2, gamma):
		linear_rbf = linear_kernel(x1, x2) + RBF_kernel(x1, x2, gamma)
		linear_rbf[:, 0] /= 2
		return linear_rbf

	y = gen_data('Y_train.csv')
	x = gen_data('X_train.csv')
	yt = gen_data('Y_test.csv')
	xt = gen_data('X_test.csv')

	C_grid_search = [2**c for c in range(-5, 16, 2)]
	Gamma_grid_search = [2**g for g in range(-15, 4, 2)]
	compare = []
	for c_try in C_grid_search:
		gamma_iter_compare = []
		for g_try in Gamma_grid_search:
			param = '-t 4 -c {} -g {}'.format(c_try, g_try)
			x_train = precomputed_kernel(x, x, g_try)
			model = svm_train(y, x_train, param)

			print("test: c = {}, g = {}".format(c_try, g_try))
			x_test = precomputed_kernel(xt, x, g_try)
			p_label, p_acc, p_val = svm_predict(yt, x_test, model)
			print(p_acc[0])
			gamma_iter_compare.append(p_acc[0])
		compare.append(gamma_iter_compare)
	print("\nCompare: ")
	print(compare)


if __name__ == '__main__':
	train_with_default_kernels()
	#train_with_linear_kernel()
	#train_with_polynomial_kernel()
	#train_with_RBF_kernel()
	#train_with_precomputed_kernel()