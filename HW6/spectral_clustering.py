from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig_idx = 0


def loadData(filepath):
	def Coordinate():
		data = [(x,y) for x in range(100) for y in range(100)]

		return data

	def Color(filepath):
		rgb_value = []

		# Read image
		img = Image.open(filepath)

		# Convert to RGB
		img_rgb = img.convert("RGB")
		for i in Coordinate():
			rgb_pixel_value = img_rgb.getpixel(i)
			rgb_value.append(rgb_pixel_value)

		return rgb_value

	return np.array(Coordinate()), np.array(Color(filepath))


def dist(a, b, ax=1):
	""" Calculate Euclidean distance """

	return np.linalg.norm(a - b, axis=ax)


def precomputed_kernel(x_s1, x_s2, x_c1, x_c2, gamma):
	def RBF_kernel(x1, x2, gamma):
		# k(x1, x2) = e^(-gamma * ||x1-x2||^2)
		n1 = x1.shape[0]
		n2 = x2.shape[0]
		K_train = np.zeros((n1, n2))
		x1_norm = np.sum(x1**2, axis=-1)
		x2_norm = np.sum(x2**2, axis=-1)
		if isinstance(x1_norm, np.ndarray) == False:
			x1_norm = np.array([x1_norm])
		if isinstance(x2_norm, np.ndarray) == False:
			x2_norm = np.array([x2_norm])

		dist = x1_norm[:, None] + x2_norm[None, :] - 2 * np.dot(x1, x2.T)
		K_train = np.exp(-gamma * dist)

		return K_train

	return RBF_kernel(x_s1, x_s2, gamma) * RBF_kernel(x_c1, x_c2, gamma)


def kernel_dist(Xj_s, Xj_c, X, clusters, k, term3, gamma):
	"""
	Calculate Kernel Distance
	k(Xj, Xj) - 2 / |Ck| * sum(A_kn * k(Xj, Xn)) + 1 / |Ck|^2 * sum(sum(A_kp * A_kq * k(Xp, Xq)))
	If data point Xn is assigned to the k-th cluster, A_kn = 1
	"""

	# Kernel distance from Xj to each centroid
	dist = np.zeros((k))

	# First term
	term1 = precomputed_kernel(Xj_s, Xj_s, Xj_c, Xj_c, gamma)

	# Second term
	cluster_cnt = np.array([(clusters == i).sum() for i in range(k)])
	term2 = np.zeros((k))
	for i in range(k):
		term2[i] = 2 / (cluster_cnt[i]) * np.sum(precomputed_kernel(Xj_s, X[0][clusters == i], Xj_c, X[1][clusters == i], gamma))
	
	return term1 - term2 + term3


def init_centroid(X, k):
	""" Return centroid in each clusters (there are k clusters) """

	C = X[np.random.choice(X.shape[0], k, replace=False), :]

	return C


def visualization(X, k, clusters, C):
	global fig_idx
	colors = ['r', 'g', 'b', 'y', 'c', 'm']
	global ax
	ax.clear()
	for i in range(k):
		points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
		ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	# ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
	plt.savefig("spectral_eig_" + str(fig_idx) + ".png")
	fig_idx += 1
	plt.draw()
	plt.pause(0.5)


def k_means(data_point, X, C, k):
	# Store the value of centroids when it updates
	C_old = np.zeros(C.shape)
	# Use 1d array (clusters) to store cluster of each point
	clusters = np.zeros(len(X))
	# Distance between new centroids and old centroids
	error = dist(C, C_old, None)

	# Loop until error = zero
	while error != 0:
		# Assign each value to the closest cluster
		new_clusters = np.zeros(clusters.shape)
		for i in range(len(X)):
			distances = dist(X[i], C)
			cluster = np.argmin(distances)
			new_clusters[i] = cluster
		clusters = new_clusters
		# Store the old centroid values
		C_old = C.copy()
		# Find new centroids by means of each cluster
		for i in range(k):
			points = X[clusters == i]
			C[i] = np.mean(points, axis=0)
		error = dist(C, C_old, None)
		print("error: ", error)

		# Visualize
		visualization(data_point, k, clusters, C)


def similarity_graph(data):
	gamma = 30
	n = data.shape[0]
	G = precomputed_kernel(data[0], data[0], data[1], data[1], gamma)

	return G


def laplacian(W):
	n = G.shape[0]

	def degree_matrix():
		# D = np.zeros((n, n))
		# for i in range(n):
		#     for j in range(n):
		#         if (i != j):
		#             D[i][j] += W[i][j]
		return np.diag(np.sum(W, axis=1))

	return np.eye(n) - degree_matrix() @ W @ degree_matrix()


def find_k_smallest_eigenvalues(L, K):
	eigen_value, eigen_vector = np.linalg.eig(L)
	sorting_index = np.argsort(eigen_value)
	eigen_value = eigen_value[sorting_index]
	eigen_vector = eigen_vector.T[sorting_index]
	print("eigen value:")
	print(eigen_value)
	print("-----------------")
	print('eigen vector:')
	print(eigen_vector)
	print("------------------")

	return eigen_value[1:k + 1], (eigen_vector[0:k])


if __name__ == '__main__':
	k = 2
	data_point = loadData("image1.png")
	plt.ion()

	# Construct a similarity graph G
	G = similarity_graph(data_point[1])
	print(G)
	# Compute the unnormalized Laplacian L
	L = laplacian(G)
	print(L)
	# Compute the first k eigenvectors u1, . . . ,uk of L
	eigen_value, eigen_vector = find_k_smallest_eigenvalues(L, k)
	print('eigen_value:')
	print(eigen_value)
	print('eigen_vector:')
	print(eigen_vector[0])
	# Cluster the points yi, where i = 1, ... , n in Rk with the k-means algorithm into clusters C1, ... , Ck
	C = init_centroid(eigen_vector.T, k)
	print(C)
	k_means(data_point, eigen_vector.T, C, k)
	plt.show()