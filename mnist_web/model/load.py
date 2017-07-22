
"""
Load the trained model
@author: Shree Ranga Raju
"""

# import libraries
import keras
import keras.models
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow
from keras.models import model_from_json

def init():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print "Loaded model from the disk"

    # compile the newly loaded model
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.adam(),
        metrics =['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph
