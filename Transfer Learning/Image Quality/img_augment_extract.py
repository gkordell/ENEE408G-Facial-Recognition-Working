# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:49:30 2018

@author: pydea
"""

import scipy.io as sio
import imageio as imgio
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dlib
import matplotlib.pyplot as plt

# This function returns the K augmented images of just the face area, as well
# as the extracted dlib features of these augmented images
# augmentation_num is the number of augmented images to create

def augment_and_extract_features(new_image_filename,augmentation_num):
    
    ## FIND THE FACE IN THE INPUT IMAGE ---------------------------------------------------------------------
    
    #new_image_filename = './new_img/Aaron_Sorkin/Aaron_Sorkin_0001.jpg'
    #Predictor path is for finding the face in the image -- will skip this step since we are using cropped images
    predictor_path = './dlib_models/shape_predictor_5_face_landmarks.dat'
    facerec_path = './dlib_models/dlib_face_recognition_resnet_model_v1.dat'
    
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    
    # Detect the faces in the image
    im = imgio.imread(new_image_filename)
    dets = detector(im, 1)
    if len(dets) == 0:  # If it detects no faces, default to using the whole thing
        d = dlib.rectangle(0,0,200,200)
    else:               # If it detects a face, get the shape and just do augmentation on the first detected face
        d = dets[0]
    
    # Record the coordinates of where the face is
    shape = np.asarray([d.top(), d.bottom(), d.left(), d.right()])
    
    # now downsize the image to the recovered coordinates
    im_just_face = np.asarray(im[shape[0]:shape[1],shape[2]:shape[3]])
    
    im_just_face = np.reshape(im_just_face, [1, im_just_face.shape[0], im_just_face.shape[1], im_just_face.shape[2]])
    
    ## AUGMENT THE INPUT IMAGE ---------------------------------------------------------------------
    
    train_datagen_obj = ImageDataGenerator(
          rescale=1./255,
          rotation_range=20,
          width_shift_range=0.2,
          height_shift_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
    
    train_generator = train_datagen_obj.flow(
            im_just_face,
            batch_size=64)
     
    aug_results = np.zeros([augmentation_num, 1, im_just_face.shape[1], im_just_face.shape[2], 3])
    
    for i in range(augmentation_num):
        temp = train_generator.next()
        aug_results[i,:,:,:,:] = temp
    
    aug_results = np.squeeze(aug_results)
    
    ## RUN DLIB ON THE AUGMENTED RESULTS ---------------------------------------------------------------------
    f = np.zeros(128)
    facerec = dlib.face_recognition_model_v1(facerec_path)
    aug_results_features = np.zeros([128,augmentation_num])
    d = dlib.rectangle(0,0,im_just_face.shape[1],im_just_face.shape[2]) 
    for i in range(augmentation_num):
        shape = sp(aug_results[i,:,:,:], d)
        face_desc = facerec.compute_face_descriptor(aug_results[i,:,:,:], shape)
        # save this face descriptor object in a numpy array
        for j in range(128):
            f[j] = face_desc[j]
        # append to features array, same index as the original array
        aug_results_features[:,i] = f
        
    return aug_results, aug_results_features
