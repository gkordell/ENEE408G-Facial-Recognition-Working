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

#def augment_and_extract_features(image_filename):
    
## FIND THE FACE IN THE INPUT IMAGE

image_parent_dir = './new_img/'
image_filename = './new_img/Aaron_Sorkin/Aaron_Sorkin_0001.jpg'
#Predictor path is for finding the face in the image -- will skip this step since we are using cropped images
predictor_path = './dlib_models/shape_predictor_5_face_landmarks.dat'
facerec_path = './dlib_models/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Detect the faces in the image
im = imgio.imread(image_filename)
dets = detector(im, 1)
if len(dets) == 0:  # If it detects no faces, default to using the whole thing
    d = dlib.rectangle(0,0,200,200)
else:               # If it detects a face, get the shape and just do augmentation on the first detected face
    d = dets[0]

# Record the coordinates of where the face is
shape = np.asarray([d.top(), d.bottom(), d.left(), d.right()])

# now downsize the image to the recovered coordinates
im_just_face = np.asarray(im[shape[0]:shape[1],shape[2]:shape[3]])

# Now re-save the image as just the face area
imgio.imwrite(image_filename,im_just_face,'JPG')


## AUGMENT THE INPUT IMAGE ---------------------------------------------------------------------


# Use batchsize parameter of 64 
num_target = 20  # create 20 augmented images from each input image
image_size = 200 # using 200x200 images

train_datagen_obj = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen_obj.flow(
        image_filename,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
 
aug_results = np.zeros([num_target,200,200,3])

for i in range(20):
    temp = train_generator.next()
    aug_results[i,:,:,:] = temp[0][1]

