import numpy as np
import matplotlib.pyplot as plt

def LU_decomposition(m):
	n = m.shape[0]
	l = np.zeros(m.shape)
	u = np.zeros(m.shape)

	for i in range(n):
		# Upper Triangular
		for j in range(i, n):
			sum = 0.0
			for k in range(i):
				sum += l[i][k] * u[k][j]
			u[i][j] = m[i][j] - sum
		# Lower Triangular
		for j in range(i, n):
			if (i == j):
				# main pivot 1
				l[i][i] = 1
			else:
				sum = 0.0
				for k in range(i):
					sum += l[j][k] * u[k][i]
				l[j][i] = (m[j][i] - sum) / u[i][i]

	return l, u


def LU_inverse(l, u):   # inverse of LU
	# LUx = I => Ly = I => y = Ux
	n = l.shape[0]
	I = np.eye(n)
	y = np.zeros(l.shape)
	x = np.zeros(l.shape)

	# solve Ly = I: backward-substitution
	for i in range(n):
		for j in range(n):
			sum = 0.0
			for k in range(j):
				sum += l[j][k] * y[k][i]
			y[j][i] = I[i][j] - sum

	# solve Ux = y: forward-substitution
	for i in range(n):
		for j in range(n-1, -1, -1):
			sum = 0.0
			for k in range(j+1, n):
				sum += u[j][k] * x[k][i]
			x[j][i] = (y[j][i] - sum) / u[j][j]

	return x


def LSE(A, lambd, b):   # (A^T * A + lambda * I)^-1 * (A^T * b)
	ATA = np.matmul(np.transpose(A), A)
	l, u = LU_decomposition(ATA + lambd * np.eye(ATA.shape[0]))
	A_inverse = LU_inverse(l, u)

	return np.matmul(np.matmul(A_inverse, np.transpose(A)), b)


def gradient(A, b, x):  # 2 * A^T * A * x - 2 * A^T * b
	return 2 * np.matmul(np.matmul(np.transpose(A), A), x) - 2 * np.matmul(np.transpose(A), b)


def hessian(A, b, x):   # 2 * A^T * A
	return 2 * np.matmul(np.transpose(A), A)


def NewtonMethod(A, b, x):
	gradient_x = gradient(A, b, x)
	hessian_x = hessian(A, b, x)

	l, u = LU_decomposition(hessian_x)
	hessian_x_inverse = LU_inverse(l, u)

	# h = f'(x) / f''(x) = Hf(x)^-1 * Gf(x)
	h = np.matmul(hessian_x_inverse, gradient_x)
	while np.linalg.norm(h) >= 0.0001:
		gradient_x = gradient(A, b, x)
		hessian_x = hessian(A, b, x)

		l, u = LU_decomposition(hessian_x)
		hessian_x_inverse = LU_inverse(l, u)
		# h = f'(x) / f''(x) = Hf(x)^-1 * Gf(x)
		h = np.matmul(hessian_x_inverse, gradient_x)

		# x(i+1) = x(i) - f'(x) / f''(x)
		x = x - h

	return x


def generateData(filepath, n_base):
	# parsing input data: x,y => Ax = b
	A = []
	b = []
	x = []
	y = []
	with open(filepath, 'r') as fp:
		for line in fp:
			data = line.split(',')
			# x
			x.append(float(data[0]))
			y.append(float(data[1]))
			# b: [y0,y1,y2...]^T
			b.append([float(data[1])])
			# A: # [x0,x1,x2...]^T
			A.append([np.power(float(data[0]), i) for i in reversed(range(n_base))])

	# end of parsing
	A = np.array(A)
	b = np.array(b)
	x = np.array(x)
	y = np.array(y)

	return A, b, x, y


if __name__ == '__main__':
	n = 3
	lambd = 10000

	### generate A, b ###
	A, b, x, y = generateData("testfile.txt", n)

	### LSE ###
	LSE_result = LSE(A, lambd, b)
	LSE_error = np.power(np.linalg.norm(np.matmul(A, LSE_result) - b), 2)

	output =  str(float(LSE_result[0])) + "X^" + str(n-1)
	for i in range(1, n):
		if (i != n-1):
			if (float(LSE_result[i]) > 0):
				output += " + " + str(float(LSE_result[i])) + "X^" + str(n-i-1)
			else:
				output += " " + str(float(LSE_result[i])) + "X^" + str(n-i-1)
		else:
			if (float(LSE_result[i]) > 0):
				output += " + " + str(float(LSE_result[i]))
			else:
				output += " " + str(float(LSE_result[i]))
	print("LSE:")
	print("Fitting line: " + output)
	print("Total error:", LSE_error)
	print("\n")

	### Newton's Method ###
	x0 = np.zeros((n, 1))
	Newton_result = NewtonMethod(A, b, x0)
	Newton_error = np.power(np.linalg.norm(np.matmul(A, Newton_result) - b), 2)

	output =  str(float(Newton_result[0])) + "X^" + str(n-1)
	for i in range(1, n):
		if (i != n-1):
			if (float(Newton_result[i]) > 0):
				output += " + " + str(float(Newton_result[i])) + "X^" + str(n-i-1)
			else:
				output += " " + str(float(Newton_result[i])) + "X^" + str(n-i-1)
		else:
			if (float(Newton_result[i]) > 0):
				output += " + " + str(float(Newton_result[i]))
			else:
				output += " " + str(float(Newton_result[i]))
	print("Newton's Method:")
	print("Fitting line: " + output)
	print("Total error:", Newton_error)

	### Visualization ###
	# LSE:
	x_LSE = np.linspace(min(x) - 1.0, max(x) + 1.0, 1000)
	y_LSE = 0
	for i in range(n):
		y_LSE += LSE_result[i] * np.power(x_LSE, n-i-1)
	# Newton:
	x_Newton = x_LSE
	y_Newton = 0
	for i in range(n):
		y_Newton += Newton_result[i] * np.power(x_Newton, n-i-1)

	plt.subplot(2, 1, 1)
	plt.plot(x, y, 'ro')
	plt.plot(x_LSE, y_LSE)
	plt.ylabel("LSE")

	plt.subplot(2, 1, 2)
	plt.plot(x, y, 'ro')
	plt.plot(x_Newton, y_Newton)
	plt.ylabel("Newton")
	plt.show()