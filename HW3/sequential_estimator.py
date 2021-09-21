import numpy as np

np.random.seed(5)

def gaussianDataGenerator(mean, variance):
	# Applying Box-Muller Method

	U = np.random.uniform(0.0, 1.0)
	V = np.random.uniform(0.0, 1.0)
	standard_norm = (-2*np.log(U))**(0.5) * np.cos(2*np.pi * V)		# standard Gaussian
	data = standard_norm * np.sqrt(variance) + mean		# standard norm = (data - mean) / std

	return data


def sequentialEstimator(m, s):
	# Applying Welford's online algorithm
	
	n = 1
	new_data = gaussianDataGenerator(m, s)
	sample_mean = new_data / n
	sample_var = 0.0

	prev_mean = sample_mean
	prev_var = sample_var

	while True:
		n += 1
		new_data = gaussianDataGenerator(m, s)
		sample_mean = prev_mean + (new_data-prev_mean)/n
		sample_var = prev_var + (new_data-prev_mean)**2 / n - prev_var / (n-1)
		print("Add data point:", new_data)
		print("Mean =", sample_mean,  "\tVariance =", sample_var)

		# check converge
		if (abs(sample_mean - m) < 0.03 and abs(sample_var - s) < 0.03):
			break

		# update previous mean and variance
		prev_mean = sample_mean
		prev_var = sample_var


if __name__ == '__main__':
	m = 3.0
	s = 5.0
	print("Data point source function: N(", m, ", ", s, ")", sep="")
	print("\n")
	sequentialEstimator(m, s)