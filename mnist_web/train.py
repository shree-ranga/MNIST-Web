
"""
Convnet to classify MNIST data set in Keras
@author: Shree Ranga Raju
"""

# import libraries
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
# import matplotlib.pyplot as plt

# Load the dataset
# data shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_train_samples = x_train.shape[0]
num_test_samples = x_test.shape[0]
# print "Number of training samples: {}".format(num_train_samples)
# print "Number of test samples: {}".format(num_test_samples)

# shape of the image
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]

# number of classes
num_classes = len(np.unique(y_train))
# print "Total number of classes in the data set: {}".format(num_classes)

# Pre-processing
# reshape train and test images
# image_data_format = "channels_last"
x_train = x_train.reshape(num_train_samples, img_rows, img_cols, 1) # 4D tensor?
x_test = x_test.reshape(num_test_samples, img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
input_shape = (img_rows, img_cols, 1)
# print "Shape of the image is {}".format(input_shape)

# Normalize
x_train /= 255
x_test /= 255

# convert class labels to binary class labels
# 9 in y_train/y_test becomes [0,0,0,0,0,0,0,0,0,1]
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

batch_size = 128
epochs = 20

# Model
model = Sequential()

# layer 1 with 32 filters and 3x3 kernel size
model.add(Conv2D(32, kernel_size=(3,3),
				 activation='relu',
				 input_shape=input_shape))
# layer 2 with 64 filters and 3x3 kernel size
model.add(Conv2D(32, (3,3), activation='relu'))
# layer 3 with 64 filters and 3x3 kernel size
model.add(Conv2D(64, (3,3), activation='relu'))
# add maxpooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# add dropout layer
model.add(Dropout(0.25))
# add a flat layer to flatten since too many dimensions
model.add(Flatten())
# fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# last layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=keras.optimizers.adam(),
			  metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train,
	 	  batch_size=batch_size,
	 	  epochs=epochs,
	 	  verbose=1,
	 	  validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
# model to json
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)

# predicted weights to HDFS format
model.save_weights("model.h5")
