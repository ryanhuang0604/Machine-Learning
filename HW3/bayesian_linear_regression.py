import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

def gaussianDataGenerator(mean, variance):
	# Applying Box-Muller Method
	U = np.random.uniform(0.0, 1.0)
	V = np.random.uniform(0.0, 1.0)
	standard_norm = (-2*np.log(U))**(0.5) * np.cos(2*np.pi * V)		# standard Gaussian
	data = standard_norm * np.sqrt(variance) + mean		# standard norm = (data - mean) / std

	return data


def linearDataGenerator(n, a, w):
	x = np.random.uniform(-1.0, 1.0)
	y = 0.0
	for i in range(n):
		y += w[i] * (x ** i)
	error = gaussianDataGenerator(0, a)

	return x, y+error


class Visualize:
	def __init__(self):
		self.graph_title = ["Ground truth", "Predict result", "After 10 incomes", "After 50 incomes"]

	def groundTruth(self, n, a, W):
		x_point = np.array([])
		y_point = np.array([])
		for x in np.arange(-2, 2, 0.1):
			x_vector = np.array([np.power(x, i) for i in range(n)])
			y = W.T @ x_vector
			x_point = np.append(x_point, x)
			y_point = np.append(y_point, y)

		plt.subplot(2, 2, 1)
		plt.xlim((-2, 2))
		plt.ylim((-15, 25))
		plt.title(self.graph_title[0])
		plt.plot(x_point, y_point, '-', color='black')
		plt.plot(x_point, y_point + a, '-', color='red')
		plt.plot(x_point, y_point - a, '-', color='red')

	def predictResult(self, x_vector, y_vector, mean, cov, a, n, subplot_idx):
		x_vector = np.array(x_vector)
		x_mean = np.array([])
		predict_x = []
		predict_y = []
		var = []
		for x in np.arange(-2, 2, 0.1):
			x_mean = np.append(x_mean, x)
			predict_x.append([x**i for i in range(n)])
			predict_y.append(predict_x[-1] @ mean)
			# plot variance
			var.append(1 / a + np.array(predict_x[-1]) @ cov @ np.array(predict_x[-1]).T)

		plt.subplot(2, 2, subplot_idx)
		plt.xlim((-2, 2))
		plt.ylim((-15, 25))
		plt.title(self.graph_title[subplot_idx - 1])
		# plot data points
		plt.plot([x_vector[i][1] for i in range(len(x_vector))], y_vector, 'o', markersize=3)

		# plot mean
		plt.plot(x_mean, predict_y, '-', color='black')

		# plot variance
		predict_y = np.array(predict_y)
		var = np.array(var).reshape((40, 1))
		plt.plot(x_mean, predict_y + var, '-', color='red')
		plt.plot(x_mean, predict_y - var, '-', color='red')


def bayesianLinearRegression(b, n, a, W):
	'''
	b: precision prior(covariance inverse)
	S = inv(prior_cov)
	m = prior_mean
	'''
	graph = Visualize()
	graph.groundTruth(n, a, W)

	# repeat update prior until the posterior probability converges
	S = b * np.eye(n)
	m = np.zeros((n, 1))
	data_x = []
	data_y = []

	in_cnt = 0
	while True:
		new_data_x, new_data_y = linearDataGenerator(n, a, W)
		in_cnt += 1
		
		new_x_vector = np.array([new_data_x**i for i in range(n)])
		data_x.append(new_x_vector)
		data_y.append([new_data_y])
		new_x_vector = np.array(data_x)
		new_y_vector = np.array(data_y)

		print("Add data point ({}, {}):\n".format(new_data_x, new_data_y))
		'''
		Calculate new posterior:
		post_cov_inv = a X^T X + S
		post_mean = post_cov (a X^T y + S m)
		'''
		# calculate new posterior
		post_S = np.dot(a * np.transpose(new_x_vector), new_x_vector) + S		
		post_mean = np.dot(np.linalg.inv(post_S), (np.dot(a * np.transpose(new_x_vector), new_y_vector) + np.dot(S, m)))
		
		# calculate predictive mean and variance
		predict_mean = np.dot(new_x_vector[-1], m)
		predict_var = 1 / a + np.dot(np.dot(new_x_vector[-1], np.transpose(S)), np.transpose(new_x_vector[-1]))
		print("Posterior mean:")
		print(post_mean)
		print("\n")
		print("Posterior variance:")
		print(np.linalg.inv(post_S))
		print("\n")
		print("Predictive distribution ~ N({}, {})".format(predict_mean[0], predict_var))
		print("-----------------------------------------------------------------------")

		if in_cnt == 10:
			graph.predictResult(data_x, data_y, post_mean, np.linalg.inv(post_S), a, n, 3)

		elif in_cnt == 50:
			graph.predictResult(data_x, data_y, post_mean, np.linalg.inv(post_S), a, n, 4)

		# check convergency
		if np.linalg.norm(np.linalg.inv(post_S) - np.linalg.inv(S)) < 0.001 and np.linalg.norm(post_mean - m) < 0.001:
			break

		# update next prior as posterior
		S = post_S
		m = post_mean

	graph.predictResult(data_x, data_y, post_mean, np.linalg.inv(post_S), a, n, 2)
	


if __name__ == "__main__":
    b = 1
    n = 4
    a = 1
    W = np.array([1, 2, 3, 4])
    bayesianLinearRegression(b, n, a, W)
    plt.show()