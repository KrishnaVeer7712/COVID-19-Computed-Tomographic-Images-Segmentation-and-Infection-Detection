import os
from app import app
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.image as img
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import sys
import random
import warnings
warnings.filterwarnings('ignore')

# import cv2 
import os
from PIL import Image
import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import nibabel as nib

from pylab import rcParams

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import  Conv2D,  MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import backend as K

file_names = []
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'files[]' not in request.files:
		flash('No file part')
		return redirect(request.url)
	files = request.files.getlist('files[]')
	
	for file in files:
		if file and allowed_file(file.filename):
			global file_name
			file_name = secure_filename(file.filename)
			file_names.append(file_name)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
		#else:
		#	flash('Allowed image types are -> png, jpg, jpeg, gif')
		#	return redirect(request.url)

	return render_template('upload.html', filenames=file_names)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

output_path=""
# my_pred_path='D:\E\MtechDAIICT\ProgrammingLab\Project\Flask\python-flask-upload-display-multiple-images\static'
my_pred_path='Predicted_image.png'
@app.route('/predict', methods=['POST'])
def predict():
	if len(file_names)!=0:
		global output_path
		output_path=(os.path.join(app.config['UPLOAD_FOLDER'], file_names[0]))
		print(output_path)

	#model = keras.models.load_model('model.h5')
	# load model weights, but do not compile
	model = load_model("model.h5", compile=False)

	m=img.imread(output_path)
	# m2=m.reshape((32, 768))
	# printing the model summary 
	# model.predict(m)
	# model = pickle.load(open("unet.pickle", 'rb'))

	return render_template('upload.html', filename='static/Results/' + my_pred_path)
	# return redirect(url_for('static', filename='Results/' + my_pred_path), code=301)

if __name__ == "__main__":
    app.run()