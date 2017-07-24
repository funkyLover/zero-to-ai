# from https://www.kaggle.com/c/digit-recognizer

import pandas as pd
import numpy as np
from random import sample
import matplotlib.pyplot as plt

# 00: load train data
train_data = pd.read_csv('train.csv')

X = np.array(train_data.drop(['label'], 1))
y = np.array(train_data['label'])

# all image pixel data
images = train_data.iloc[:, 1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

# print(X[:5])
# print(y[:5])

# 01: visualize the data example randomly
def display_random(max):
	# default count to display
	count = 9
	idx = sample(range(1, max), count)
	display_images = images[idx]
	display_images = [img.reshape(28, 28) for img in display_images]
	# plt.subplots(2,5, figsize=(15, 6)

	for i, v in enumerate(display_images):
		plt.subplot(3, 3, i + 1)
		plt.axis('off')
		plt.imshow(v, cmap=plt.cm.gray_r, interpolation='nearest')

	plt.show()

display_random(len(images))
