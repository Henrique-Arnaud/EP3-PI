import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []
classes = 43
cur_path = os.getcwd()
for i in range (classes):
	path = os.path.join(cur_path, 'train', str(i))
	images = os.listdir(path)
	for a in images:
		try:
			image = Image.open(path + '\\' + a)
			image-image.resize((30,30))
			image - np.array(image)
			Image.fromrray (image)
			data.append(image)
			labels.append(i)
		except:
			print("Error loading image")
data = np.array(data)
print(data)
labels = np.array(labels)
