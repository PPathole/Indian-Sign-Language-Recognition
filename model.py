import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
predict = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
def loadmodel(filepath):
	model = tf.keras.models.load_model(filepath,custom_objects=None,compile=True)
	return model
	

def prediction(path,model):
	img = image.load_img(path, target_size=(150, 150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	images = np.vstack([x])
	classes = model.predict(images)	
	for i in range(0,24):
		if classes[0][i] == 1:
			return predict[i]
    

model = loadmodel('C://Users/Admin/Downloads/model.h5')
print(prediction('C://Users/Admin/Documents/isl project/C/001.jpg', model))