#argv[1] is path to image (unnecessary??), argv[2] is it's label

# ONLINE LEARNING STEPS SUMMARY
# 1. Get the new image of the new person, and generate augmented images from it
# 		# On the new images, and use dlib to find the face and extract the features
#         # on this part, use example from http://dlib.net/face_recognition.py.html for guidance
#         # USE THE SCRIPT feature_extract_augment.py to do steps 2-3
# 2. Load already-trained keras model weights
# 3. Re-train the keras model on the new dataset with the new face added from step 4


from keras.utils import to_categorical  
from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# from keras.optimizers import *
# from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np
import sys
# from keras.preprocessing.image import ImageDataGenerator
# import dlib

if __name__ == "main":

	#define paramters of model
	aug_num = 50
	epochs = 40

	### ONLINE LEARNING ---------------------------------------------------------
	## STEP 1 ---------------------------------------------------------------------
	#import new image, get features
	images, x_train = augment_and_extract_features(int(sys.argv[1]), aug_num)
	# x_train = np.ones([128,aug_num])*0.1
	y_train = np.ones([1, aug_num])*int(sys.argv[2]) # ? How to get label?
	# get testing data
	x_test = sio.loadmat('../Transfer Learning/dlib_features/test_features.mat')['test_features']
	y_test = sio.loadmat('../Transfer Learning/Class/test_class.mat')['test_class']

	current_num_classes = np.max(y_test) + 1   # this assumes that all classes are present in y

	x_train = np.transpose(x_train)
	x_test = np.transpose(x_test)

	y_train = to_categorical(y_train, current_num_classes)
	y_test = to_categorical(y_test, current_num_classes)

	y_train = np.squeeze(y_train)
	y_test = np.squeeze(y_test)

	## STEP 2 ---------------------------------------------------------------------
	#load pretrained model
	model = load_model('../Models/dlib_classifierV0_trained.h5')

	## STEP 3 ---------------------------------------------------------------------
	#retrain model with new face

	history = model.fit(
		x_train, 
		y_train,
		epochs = epochs,
		validation_data = (x_test, y_test))


	model.save_weights('../Models/dlib_classifierV0_trained.h5')


