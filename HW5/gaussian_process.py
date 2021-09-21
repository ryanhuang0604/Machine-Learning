import numpy as np
from scipy.spatial.distance import cdist as cdist
import matplotlib.pyplot as plt

np.random.seed(5)

def read_data(filename):
	x = []
	y = []
	result = []
	with open(filename, 'r') as f:
		for line in f:
			x.append(float(list(line.strip('\n').split(' '))[0]))
			y.append(float(list(line.strip('\n').split(' '))[1]))

	return x, y


def rationalQuadraticKernel(x1, x2, l=0.1, alpha=1.0):
	x1 = np.array(x1).reshape(2, -1)
	x2 = np.array(x2).reshape(2, -1)
	dist = (1 + cdist(x1, x2, 'euclidean')**2 / (2*alpha*l**2)) ** (-alpha)

	return dist


def GPRegression(x, f_x):
	epsilon = np.random.normal(0, 5**(-1), len(x))
	y = f_x + epsilon

	noise_var = 1e-6

	# compute mean
	N = len(x)
	K = rationalQuadraticKernel(x, x)
	L = np.linalg.cholesky(K + noise_var*np.eye(N))
	Lk = np.linalg.solve(L, K)
	mu = np.dot(Lk.T, np.linalg.solve(L, y))
	
	# compute variance
	sd = np.sqrt(np.diag(K) - np.sum(Lk**2, axis=0))
	
	return y, mu, sd


def visualization(x, y, y_pred, sigma):
	# Plot the function, prediction and 95% confidence interval
	plt.figure()
	plt.plot(x, y, 'r:', label=r'$f(x)')
	plt.plot(X, y, 'r.', markersize=10, label='Training points')
	plt.plot(x, y_pred, 'b-', label='Prediction')
	plt.fill(np.concatenate([x, x[::-1]]),
	         np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
	         alpha=.5, fc='b', ec='None', label='95% confidence interval')
	plt.xlabel('$x$')
	plt.ylabel('$f(x)$')


if __name__ == '__main__':
	x, f_x = read_data("input.data")
	y, mu, sd = GPRegression(x, f_x)
	visualization(x, f_x, y, sd)