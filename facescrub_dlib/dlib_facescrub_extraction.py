# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:46:25 2018

@author: pydea
"""

# This is the script which takes the FaceScrub Data Compressed in .mat format and does feature extraction
# results will be saved in "train_features.mat" and "test_features.mat"

import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import face_recognition
import dlib

# Import all the images from Arun's MAT files
x_train_raw_1 = sio.loadmat('./DataWithMean/training_data_1_with_mean.mat')['training_data1']
print('done loading training data 1/3')
x_train_raw_2 = sio.loadmat('./DataWithMean/training_data_2_with_mean.mat')['training_data2']
print('done loading training data 2/3')
x_train_raw_3 = sio.loadmat('./DataWithMean/training_data_3_with_mean.mat')['training_data3']
print('done loading training data 3/3')
x_train_raw = np.concatenate((x_train_raw_1,x_train_raw_2),axis = 3)
x_train_raw = np.concatenate((x_train_raw,x_train_raw_3),axis = 3)
print('done concatenation of training data')
del x_train_raw_1
del x_train_raw_2
del x_train_raw_3
#%%
x_test_raw = sio.loadmat('./DataWithMean/test_data_with_mean.mat')['test_data']
print('done loading test data')

#%%
# Now do the feature extraction using dlib library

#Predictor path is for finding the face in the image -- will skip this step since we are using cropped images
predictor_path = './dlib_models/shape_predictor_5_face_landmarks.dat'
facerec_path = './dlib_models/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(facerec_path)

# Extract the training features
length = x_train_raw.shape[3]

train_features = np.zeros([128,length])
f = np.zeros(128)
for i in range(length):
    img = x_train_raw[:,:,:,i]
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # let the dlib find the shape of where the face is (can't find a way to avoid this)
        shape = sp(img, d)
        # run the face descriptor
        face_desc = facerec.compute_face_descriptor(img, shape)
        # save this face descriptor object in a numpy array
        for j in range(128):
            f[j] = face_desc[j]
        # append to features array, same index as the original array
        train_features[:,i] = f
    print('calculating on training data ',i)

# Save as a new .mat file
train_features_dict = {}
train_features_dict['train_features'] = train_features
sio.savemat('train_features',train_features_dict)

print('Training features saved -- now moving on to test features')

#%%
# Extract the test features
length = x_test_raw.shape[3]

test_features = np.zeros([128,length])
f = np.zeros(128)
for i in range(length):
    img = x_test_raw[:,:,:,i]
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # let the dlib find the shape of where the face is (can't find a way to avoid this)
        shape = sp(img, d)
        # run the face descriptor
        face_desc = facerec.compute_face_descriptor(img, shape)
        # save this face descriptor object in a numpy array
        for j in range(128):
            f[j] = face_desc[j]
        # append to features array, same index as the original array
        test_features[:,i] = f
    print('calculating on test data ',i)
    
# Save as a new .mat file
test_features_dict = {}
test_features_dict['test_features'] = test_features
sio.savemat('test_features',test_features_dict)

print('Test features saved - done!')