import os.path
import struct as st
import numpy as np

def ImgDataGen(filepath):
	train_imagesfile = open(filepath, 'rb')

	# Read the magic number
	train_imagesfile.seek(0)
	magic = st.unpack('>4B', train_imagesfile.read(4))

	# Read the dimensions of the Image data-set
	nImg = st.unpack('>I', train_imagesfile.read(4))[0]  #num of images
	nR = st.unpack('>I', train_imagesfile.read(4))[0]  #num of rows
	nC = st.unpack('>I', train_imagesfile.read(4))[0]  #num of column

	# Reading the Image data
	nBytesTotal = nImg * nR * nC * 1  #since each pixel data is 1 byte
	images_array = np.asarray(
		st.unpack('>' + 'B' * nBytesTotal,
				  train_imagesfile.read(nBytesTotal))).reshape((nImg, nR, nC))

	# Tally into 32 bins
	images_array = images_array / 8

	# flattering images_array: 60000 * 28 * 28 => 60000 * 784
	# [3, 6] => (3-1) * 28 + 6
	images_array = images_array.reshape(
		images_array.shape[0], images_array.shape[1] * images_array.shape[2])

	train_imagesfile.close()
	return images_array


def LabelDataGen(filepath):
	train_labelfile = open(filepath, 'rb')

	# Read the magic number
	train_labelfile.seek(0)
	magic = st.unpack('>4B', train_labelfile.read(4))

	# Read the number of data-set items
	nItem = st.unpack('>I', train_labelfile.read(4))[0]  #num of items

	# Reading the Label data
	nBytesTotal = nItem * 1  #since each item is 1 byte
	labels_array = np.asarray(
		st.unpack('>' + 'B' * nBytesTotal,
				  train_labelfile.read(nBytesTotal))).reshape((nItem))

	train_labelfile.close()
	return labels_array


def readData():
	train_file = {
		'images': 'train-images-idx3-ubyte',
		'labels': 'train-labels-idx1-ubyte'
	}
	test_file = {
		'images': 't10k-images-idx3-ubyte',
		'labels': 't10k-labels-idx1-ubyte'
	}

	if os.path.exists('train_image.npy'):
		train_image_array = np.load('train_image.npy')
	else:
		train_image_array = ImgDataGen(train_file['images'])
		np.save('train_image', train_image_array)

	if os.path.exists('train_label.npy'):
		train_label_array = np.load('train_label.npy')
	else:
		train_label_array = LabelDataGen(train_file['labels'])
		np.save('train_label', train_label_array)

	# Read data from testing files
	if os.path.exists('test_image.npy'):
		test_image_array = np.load('test_image.npy')
	else:
		test_image_array = ImgDataGen(test_file['images'])
		np.save('test_image', test_image_array)

	if os.path.exists('test_label.npy'):
		test_label_array = np.load('test_label.npy')
	else:
		test_label_array = LabelDataGen(test_file['labels'])
		np.save('test_label', test_label_array)

	return train_image_array, train_label_array, test_image_array, test_label_array


def cntAppearFreq(img_array, label_array):
	# Initialize appearence frequency:
	appearFreq = np.zeros((10, 32, 784))  # 28*28 pixels, 32 bins, num 0~9

	if os.path.exists('appear_freq.npy'):
		appearFreq = np.load('appear_freq.npy')
	else:
		# Count appearence of bin of each pixel
		for i in range(60000):
			showed_num = int(label_array[i])
			for j in range(784):
				pixel_bin = int(img_array[i][j])
				appearFreq[showed_num][pixel_bin][j] += 1
		np.save('appear_freq', appearFreq)

	return appearFreq


def cntProbability(img_array, label_array):
	if os.path.exists('log_p.npy'):
		log_probability = np.load('log_p.npy')
	else:
		# Count appearence of bin of each pixel
		appearFreq = cntAppearFreq(img_array, label_array)

		# Calculate probablility:
		# appearence of each number
		num_total_appear = np.zeros(10)
		for i in range(10):
			num_total_appear[i] = np.sum(appearFreq[i])
		print(num_total_appear)
		# appearence of each bin
		for i in range(appearFreq.shape[0]):
			for j in range(appearFreq.shape[1]):
				for k in range(appearFreq.shape[2]):
					appearFreq[i][j][k] /= num_total_appear[i]

		# Scale to log:
		log_probability = np.zeros(appearFreq.shape)
		for i in range(appearFreq.shape[0]):
			for j in range(appearFreq.shape[1]):
				for k in range(appearFreq.shape[2]):
					if appearFreq[i][j][k] == 0.0:
						appearFreq[i][j][k] += 1e-9

					log_probability[i][j][k] = np.log(appearFreq[i][j][k])

		np.save('log_p', log_probability)

	return log_probability


def posterior(log_probability, test_img, test_label):
	# Calculate Likelihood:
	Likelihood = np.zeros(10)
	for i in range(10):
		Likelihood[i] = 0.0
		for j in range(784):
			img_bin = int(test_img[j])
			Likelihood[i] += log_probability[i][img_bin][j]

	# Calculate Prior:
	Prior = np.zeros(10)
	for i in range(10):
		# A/A+B, B/A+B
		# logA/logA+logB  = logA/logAB
		Prior[i] = np.sum(np.exp(
			log_probability[i]))  # / np.sum(log_probability)
		# Prior[i] = 0
	Prior /= np.sum(Prior)

	# Calculate Posterior:
	posterior = np.zeros(10)
	for i in range(10):
		posterior[i] = Likelihood[i] + Prior[i]
		#print("Prior: ")
		#print(Prior[i])

	posterior = posterior / np.sum(posterior)

	return np.argmin(posterior), posterior


def showBayse(img_array, label_array):
	appearFreq = cntAppearFreq(img_array, label_array)  # (10, 32, 784)
	bin_num = appearFreq.shape[1]
	pixel_num = appearFreq.shape[2]

	print("Imagination of numbers in Bayesian classifier:")
	for i in range(10):
		print('\n', i, ':')
		white = np.sum(appearFreq[i][0:16], axis=0)
		black = np.sum(appearFreq[i][16:], axis=0)
		for j in range(pixel_num):
			print(0, end='') if white[j] > black[j] else print(1, end='')
			if (j + 1) % 28 == 0:
				print('')


def showPosterior(posterior):
	print("Posterior (in log scale):")
	for i in range(10):
		print(i, ':', posterior[i])


def discreteBayse(train_image_array, train_label_array, test_image_array,
				  test_label_array):

	# Calculate Probability
	probablility = cntProbability(train_image_array,
								  train_label_array)  # (10, 32, 784)

	# Calculate Posterior
	cnt = 0
	data_size = test_image_array.shape[0]
	for i in range(data_size):
		test_posterior, posterior_list = posterior(
			probablility, test_image_array[i], test_label_array[i])
		test_label = test_label_array[i]
		showPosterior(posterior_list)
		print('Prediction: ', test_posterior, ', Ans: ', test_label)
		print('\n')
		if test_posterior == test_label:
			cnt += 1

	showBayse(train_image_array, train_label_array)

	accuracy = float(cnt / data_size)
	print("Error rate: ", 1.0 - accuracy)


def LabelCnt(img_array, label_array):
	sample_num = img_array.shape[0]
	pixel_num = img_array.shape[1]

	label_cnt = np.zeros((10))
	if os.path.exists('label_cnt.npy'):
		label_cnt = np.load('label_cnt.npy')
	else:
		for i in range(sample_num):  # 60000
			label = int(label_array[i])
			label_cnt[label] += 1
		np.save('label_cnt', label_cnt)

	return label_cnt


'''
def summationX(img_array, label_array):
	sample_num = img_array.shape[0]
	pixel_num = img_array.shape[1]

	sum_x = np.zeros((10, pixel_num))
	if os.path.exists('summation_x.npy'):
		sum_x = np.load('summation_x.npy')
	else:
		for i in range(sample_num):  # 60000
			label = int(label_array[i])
			for j in range(pixel_num):  # 784
				sum_x[label][j] += img_array[i][j]
		np.save('summation_x', sum_x)

	return sum_x
'''


def trainMean(img_array, label_array):
	sample_num = img_array.shape[0]
	pixel_num = img_array.shape[1]

	# count label L on pixel P's appearance
	train_mean = np.zeros((10, 784))  # 10 * 784
	if os.path.exists('train_mean.npy'):
		train_mean = np.load('train_mean.npy')

	else:
		label_cnt = np.zeros((10))
		for i in range(sample_num):  # 60000
			label = int(label_array[i])
			label_cnt[label] += 1
			for j in range(pixel_num):  # 784
				train_mean[label][j] += img_array[i][j]

		for i in range(10):
			for j in range(pixel_num):
				train_mean[i][j] /= label_cnt[i]
		np.save('train_mean', train_mean)
	# print(max(train_mean[5]))
	return train_mean


def trainVariance(img_array, label_array):
	sample_num = img_array.shape[0]
	pixel_num = img_array.shape[1]

	mean = trainMean(img_array, label_array)
	variance = np.zeros(mean.shape)
	if os.path.exists('train_var.npy'):
		variance = np.load('train_var.npy')
	else:
		label_cnt = np.zeros((10))
		for i in range(sample_num):  # 60000
			label = int(label_array[i])
			label_cnt[label] += 1
			for j in range(pixel_num):  #784
				variance[label][j] += np.power(
					img_array[i][j] - mean[label][j], 2)

		for i in range(10):
			variance[i] /= label_cnt[i]
		np.save('train_var', variance)

	# print(max(variance[1]))
	return variance


def gaussian_pdf(x, mean, variance):
	"""
		log form
	"""
	#print(variance)
	return np.log(1 / np.sqrt(2 * np.pi * variance)) + (-1) * np.power(
		(x - mean), 2) / (2 * variance)


def continuousPosterior(single_img, label_array, train_mean, train_var,
						label_cnt):
	# Calculate Likelihood:
	pixel_cnt = single_img.shape[0]  # 784
	single_img *= 8
	likelihood = np.zeros((10))
	for i in range(10):
		x = np.max(train_var[i] / 10)
		train_var[i] += (x)
		for j in range(pixel_cnt):  # 784

			#if train_var[i][j] == 0.0:
			#    train_var[i][j] += 1e-9
			gpdf = gaussian_pdf(single_img[j], train_mean[i][j],
								train_var[i][j] + 1e-6)
			likelihood[i] += gpdf
		train_var[i] -= x
	# Calculate Prior:
	prior = np.zeros((10))
	total_label_cnt = np.sum(label_cnt)
	for i in range(10):
		prior[i] = label_cnt[i] / total_label_cnt

	# Calculate Posterior:
	posterior = np.zeros((10))
	for i in range(10):
		posterior[i] = likelihood[i] + np.log(prior[i])

	posterior = posterior / np.sum(posterior)

	return np.argmin(posterior), posterior


def continuousBayse(train_image_array, train_label_array, test_image_array,
					test_label_array):
	train_mean = trainMean(train_image_array, train_label_array)
	train_variance = trainVariance(train_image_array, train_label_array)

	label_cnt = LabelCnt(train_image_array, train_label_array)

	cnt = 0
	data_size = test_image_array.shape[0] // 200
	for i in range(data_size):
		test_posterior, posterior_list = continuousPosterior(
			test_image_array[i], test_label_array[i], train_mean,
			train_variance, label_cnt)
		showPosterior(posterior_list)
		print('Prediction: ', test_posterior, ', Ans: ', test_label_array[i])
		print('\n')
		if test_posterior == test_label_array[i]:
			cnt += 1

	showBayse(train_image_array, train_label_array)
	accuracy = float(cnt / data_size)
	print("Error rate: ", 1.0 - accuracy)


if __name__ == '__main__':
	# choose mode: 0(discrete), 1(continous)
	usr_mode = 1

	# Open the IDX file in readable binary mode.
	train_image_array, train_label_array, test_image_array, test_label_array = readData(
	)

	# Naive Bayse's Classifier:
	if usr_mode == 0:
		# Tally into 32 bins
		# train_image_array = train_image_array / 8
		discreteBayse(train_image_array, train_label_array, test_image_array,
					  test_label_array)
	else:
		continuousBayse(train_image_array, train_label_array, test_image_array,
						test_label_array)