import numpy as np
import math
import scipy.stats as ss
import matplotlib.pyplot as plt

def Psuccess(data):
	success = data.count('1')
	total = len(data)
	return success, total, success/total


def Combination(N, m):
	return math.factorial(N) / (math.factorial(N - m) * math.factorial(m))


def Beta(a, b):
	return math.factorial(a+b-1) / (math.factorial(a-1) * math.factorial(b-1))


def Likelihood(p, N, m):
	return Combination(N, m) * np.power(p, m) * np.power(1-p, N-m)


if __name__ == '__main__':
	filename = "testfile.txt"
	a = 0
	b = 0
	x = np.linspace(0, 1, 5000)

	with open(filename) as fp:
		case = 1
		for line in fp:
			line = line.rstrip()
			success, total, p_success = Psuccess(line)
			likelihood = Likelihood(p_success, total, success)

			# prior
			y = ss.beta.pdf(x, a, b)
			plt.subplot(1, 2, 1)
			plt.plot(x, y)
			plt.annotate("prior", xy=(0, max(y)))

			'''
			# likelihood
			t1 = success
			t2 = total - success
			plt.subplot(1, 3, 2)
			plt.plot(t1, t2)
			plt.annotate("likelihood", xy=(0, max(y)))
			'''
			
			a = success + a
			b = total - success + b

			# posterior
			y = ss.beta.pdf(x, a, b)
			plt.subplot(1, 2, 2)
			plt.plot(x, y)
			plt.annotate("posterior", xy=(0, max(y)))

			plt.subplots_adjust(wspace=0.5)
			plt.show()
			
			case += 1