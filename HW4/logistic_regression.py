import numpy as np
import matplotlib.pyplot as plt

def gaussianDataGenerator(mean, variance):
	# Applying Box-Muller Method
	U = np.random.uniform(0.0, 1.0)
	V = np.random.uniform(0.0, 1.0)
	standard_norm = (-2*np.log(U))**(0.5) * np.cos(2*np.pi * V)		# standard Gaussian
	data = standard_norm * np.sqrt(variance) + mean		# standard norm = (data - mean) / std

	return data


def generateDataPoints(N, m_x, v_x, m_y, v_y):
	data_points_x = np.zeros((N, 3))

	for i in range(N):
		data_points_x[i][0] = gaussianDataGenerator(m_x, v_x)
		data_points_x[i][1] = gaussianDataGenerator(m_y, v_y)
		data_points_x[i][2] = 1

	return data_points_x


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def gradientDescent(data_x, data_y):
	N = data_x.shape[0]
	w = np.zeros((3, 1), dtype='float64')
	design_matrix_x = data_x
	last_w = w.copy()
	learning_rate = 0.001
	while (True):
		# w_n+1 = w_n + X^T * ( yi - 1 / (1+e ^ (-Xi * w)) )
		gradient = design_matrix_x.T @ (data_y - sigmoid(design_matrix_x @ w))
		# 加上一個常數 (learning rate)，來加速逼近的速度
		w = w + learning_rate * gradient
		if (np.linalg.norm(w - last_w) <= 0.01):
			break
		last_w = w.copy()

	print("Gradient Descent:\n")
	print_result(w, design_matrix_x @ w, data_y)


	return design_matrix_x @ w


def newtonHessian(x, y, w):
	N = x.shape[0]
	# D_ii = e ^ (-Xi * w) / (1 + e ^ (-Xi * w)) ^ 2
	diagonal_pivot = np.exp(-x @ w) / ((1 + np.exp(-x @ w))**2)
	diagonal_pivot = diagonal_pivot.reshape(N)
	D = np.diag(diagonal_pivot)

	return x.T @ D @ x


def newtonMethod(data_x, data_y):
	N = data_x.shape[0]
	design_matrix_x = data_x
	w = np.zeros((3, 1), dtype='float64')
	last_w = w.copy()

	while (True):
		gradient = design_matrix_x.T @ (data_y - sigmoid(design_matrix_x @ w))
		# 若遇到 Hessian matrix 無法做 inverse 時，則退回 logistic regression 假設一個 learning rate
		try:
			hessian_inverse = np.linalg.inv(newtonHessian(design_matrix_x, data_y, w))
		except np.linalg.LinAlgError:
			w = w + gradient
		# w_n+1 = w_n + H^-1 * gradient
		else:
			w = w + hessian_inverse @ gradient
		if (np.linalg.norm(w - last_w) <= 0.01):
			break
		last_w = w.copy()

	print("Newton's method:\n")
	print_result(w, design_matrix_x @ w, data_y)

	return design_matrix_x @ w


class Visualization:
	def __init__(self):
		self.graph_title = ["Ground truth", "Gradient descent", "Newton's method"]

	def draw(self, data_points, predict_label, subplot_idx):
		N = data_points.shape[0]
		plt.subplot(1, 3, subplot_idx + 1)
		plt.title(self.graph_title[subplot_idx])

		data_x = data_points[:, 0].reshape(N, 1)
		data_y = data_points[:, 1].reshape(N, 1)
		# cluster 1
		cluster_1_data_x = data_x[predict_label <= 0.5]
		cluster_1_data_y = data_y[predict_label <= 0.5]
		# cluster 2
		cluster_2_data_x = data_x[predict_label > 0.5]
		cluster_2_data_y = data_y[predict_label > 0.5]
		# plot
		plt.plot(cluster_1_data_x, cluster_1_data_y, '.', color='red')
		plt.plot(cluster_2_data_x, cluster_2_data_y, '.', color='blue')


def print_result(w, predict_label, ground_truth):
	N = predict_label.shape[0]
	print("w:")
	print(w[0][0], w[1][0], w[2][0], sep="\n")
	confusion_matrix = np.zeros((2, 2))

	for i in range(N):
		if (predict_label[i] <= 0.5 and ground_truth[i] <= 0.5):
			confusion_matrix[0][0] += 1
		elif (predict_label[i] > 0.5 and ground_truth[i] <= 0.5):
			confusion_matrix[0][1] += 1
		elif (predict_label[i] <= 0.5 and ground_truth[i] > 0.5):
			confusion_matrix[1][0] += 1
		elif (predict_label[i] > 0.5 and ground_truth[i] > 0.5):
			confusion_matrix[1][1] += 1

	'''
	confusion_matrix_result = pandas.DataFrame(confusion_matrix, columns=["Predict cluster 1", "Predict cluster 2"], index=["Is cluster 1", "Is cluster 2"])
	print("\n")
	print("Confusion Matrix:")
	print(confusion_matrix_result)
	'''
	print("\n")
	print("Confusion Matrix:")
	print("\t\tPredict cluster 1\tPredict cluster 2")
	print("Is cluster 1\t\t", int(confusion_matrix[0][0]), "\t\t\t", int(confusion_matrix[0][1]), sep="")
	print("Is cluster 2\t\t", int(confusion_matrix[1][0]), "\t\t\t", int(confusion_matrix[1][1]), sep="")
	
	print("\nSensitivity (Successfully predict cluster 1):", confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1]))
	print("Specificity (Successfully predict cluster 2):", confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]))


def logisticRegression(N, m_x1, v_x1, m_y1, v_y1, m_x2, v_x2, m_y2, v_y2):
	data_points_1 = generateDataPoints(N, m_x1, v_x1, m_y1, v_y1)
	data_points_2 = generateDataPoints(N, m_x2, v_x2, m_y2, v_y2)
	data_label_1 = np.zeros((N, 1))
	data_label_2 = np.ones((N, 1))
	data_points = np.concatenate((data_points_1, data_points_2))
	data_label = np.concatenate((data_label_1, data_label_2))

	# ground truth
	graph = Visualization()
	graph.draw(data_points, data_label, 0)

	# gradient descent
	gradient_predict = gradientDescent(data_points, data_label)
	graph.draw(data_points, gradient_predict, 1)

	print("\n----------------------------------------------------------")

	# Newton's method
	newton_predict = newtonMethod(data_points, data_label)
	graph.draw(data_points, newton_predict, 2)

	plt.show()


if __name__ == '__main__':
	N = 50
	m_x1 = 1
	v_x1 = 2
	m_y1 = 1
	v_y1 = 2
	m_x2 = 3
	v_x2 = 4
	m_y2 = 3
	v_y2 = 4
	
	logisticRegression(N, m_x1, v_x1, m_y1, v_y1, m_x2, v_x2, m_y2, v_y2)