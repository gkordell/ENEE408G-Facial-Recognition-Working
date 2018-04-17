# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:46:45 2018

@author: pydea
"""

"""
TRANSFER LEARNING STEPS SUMMARY
1. Load the known feature-extracted dataset via sio.loadmat('dlib_features/train_features.mat')['train_features']
2. Get the new image of the new person, and generate augmented images from it
3. On the new images, and use dlib to find the face and extract the features
        # on this part, use example from http://dlib.net/face_recognition.py.html for guidance
        # USE THE SCRIPT feature_extract_augment.py to do steps 2-3
4. Add subset of the new face features to train and test sets from step 1
5. Load already-trained keras model weights
6. Increase the size of the output layer of the keras model by 1 (assuming adding 1 person)
7. Re-train the keras model on the new dataset with the new face added from step 4

DIFFERENCES FOR ONLINE LEARNING:
 - Still have to do steps 2-3 above
 - Don't have to do step 1
 - Don't have to do step 4
 - Don't have to do step 6
"""


from keras.utils import to_categorical  
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import *
from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dlib

#define paramters of model
input_dim = 128
num_classes = 530
batch_size = 64
epochs = 35


## STEP 1 ---------------------------------------------------------------------
#import data
x_train = sio.loadmat('dlib_features/train_features.mat')['train_features']
x_test = sio.loadmat('dlib_features/test_features.mat')['test_features']
y_train = sio.loadmat('Class/training_class.mat')['training_class']
y_test = sio.loadmat('Class/test_class.mat')['test_class']
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = np.reshape(x_train,[36213,128])
x_test = np.reshape(x_test,[15519,128])
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)




