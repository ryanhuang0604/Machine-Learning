import numpy as np

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


if __name__ == '__main__':
	data = gaussianDataGenerator(0, 1)
	print("data:", data)
	print("\n")

	x, y = linearDataGenerator(3, 0.3, [1, 2, 3])
	print("x:", x, " y:", y)