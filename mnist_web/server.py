
"""
Server file
@author: Shree Ranga Raju
"""

# import libraries
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re

import sys
import os
sys.path.append(os.path.abspath("./model"))
from load import *

# this function decodes from base64 to raw representation
def convertImage(imgData1):
	imgstr = re.search(r'base64,(.*)', imgData1).group(1)
	with open('output.png', 'wb') as output:
		output.write(imgstr.decode('base64'))

# Initialize the app from flask
app = Flask(__name__)

# from load get default tf graph and model
# graph here is akin to a session
global model, graph
model, graph = init()

# route to main page
@app.route('/')
def index():
	return render_template('index.html')

# when user clicks submit/predict button
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
	# get the image data from the canvas
	imgData = request.get_data()
	convertImage(imgData) # comment this when debugging with postman
	# read the data
	x = imread('out.png', mode='L')
	# invert the image (black to white and viceversa) coz
	# canvas background is white but models were trained on
	# black background
	x = np.invert(x) # comment this part when debugging with postman
	# resize the image to 28x28 coz that's what the model is trained on
	x = imresize(x,(28,28))
	x = x.reshape(1,28,28,1) # This 4D tensor is fed to the model
	with graph.as_default():
		# predict the output (digit) from the model
		out = model.predict(x)
		print(np.argmax(out,axis=1))
		# convert the output to string array
		response = np.array_str(np.argmax(out, axis=1))
		return response

if __name__ == '__main__':
	# run the app on 127.0.0.1
	app.run(debug=True, port=8080)
