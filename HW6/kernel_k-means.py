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


def init_centroid(X, k, init_type="kmeans-plus"):
	""" Return centroid in each clusters (there are k clusters) """

	n = X.shape[0]
	
	if init_type == "random":
		idx = np.random.randint(len(X), size=k)
		C = X[idx]

		return C, idx

	elif init_type == "kmeans-plus":
		C = []
		C_idx = []

		# Randomly choose one to be the first centroid
		idx = np.random.randint(n)
		C.append(X[idx])
		C_idx.append(idx)

		# Use shortest distance to be the weight and find next centroid
		while len(C) < k:
			D = []
			for i in range(n):
				d_to_centroids = np.array([dist(X[i], C[j], None) for j in range(len(C))])
				D.append(np.min(d_to_centroids))
			D = np.array(D)
			D_weight = D / np.sum(D)
			next_C_idx = np.random.choice(n, p=D_weight)
			C.append(X[next_C_idx])
			C_idx.append(next_C_idx)
		C = np.array(C)
		C_idx = np.array(C_idx)

		return C, C_idx


def visualization(X, k, clusters):
	global fig_idx
	colors = ['r', 'g', 'b', 'y', 'c', 'm']
	global ax
	ax.clear()
	for i in range(k):
		points = X[clusters == i]
		ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
	#ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
	plt.savefig("kernel k-means_" + str(fig_idx) + ".png")
	fig_idx += 1
	plt.draw()
	plt.pause(0.3)


def kernel_k_means(X, C, C_idx, k):
	"""
	Kernel K-means clustering 
	Input: X(data points), C(centroid), C_idx(index of centroid), k(# of clusters)
	"""
	gamma = 8
	# gamma = 30

	# Use 1d array (clusters) to store cluster of each point
	clusters = np.repeat(-1, len(X[0]))
	for i in range(len(C_idx)):
		clusters[C_idx[i]] = i

	done = 1

	# Loop until new cluster = old cluster 
	while done:
		# Precompute kernel distance term 3
		# 1 / |Ck|^2 * sum(sum(A_kp * A_kq * k(Xp, Xq)))
		term3 = np.zeros(k)
		cluster_cnt = np.array([(clusters == i).sum() for i in range(k)])
		for p in range(k):
			result = 0
			result = np.sum(precomputed_kernel(X[0][clusters == p], X[0][clusters == p], X[1][clusters == p], X[1][clusters == p], gamma))
			term3[p] = 1 / (cluster_cnt[p]**2) * result

		# Assign each value to the closest cluster
		new_clusters = np.zeros(clusters.shape)
		for i in range(X[0].shape[0]):
			distances = kernel_dist(X[0][i], X[1][i], X, clusters, k, term3, gamma)
			cluster = np.argmin(distances)
			new_clusters[i] = cluster

		# If new cluster = last one, done
		if ((new_clusters - clusters).all()):
			done = 0
		
		clusters = new_clusters

		# Visualize
		visualization(X[0], k, clusters)


if __name__ == '__main__':
	data_point = loadData("image1.png")
	plt.ion()

	# Number of clusters
	k = 2
	# Initialize centroids
	centroid, centroid_idx = init_centroid(data_point[1], k)
	# Calculate kernel K-means
	kernel_k_means(data_point, centroid, centroid_idx, k)
	plt.show()