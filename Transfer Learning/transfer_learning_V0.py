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
8. Save the new model

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
from numpy.random import random
from keras.preprocessing.image import ImageDataGenerator
import dlib

from img_augment_extract import augment_and_extract_features

## STEP 1 ---------------------------------------------------------------------
#import data
x_train = sio.loadmat('dlib_features/train_features.mat')['train_features']
x_test = sio.loadmat('dlib_features/test_features.mat')['test_features']
y_train = sio.loadmat('Class/training_class.mat')['training_class']
y_test = sio.loadmat('Class/test_class.mat')['test_class']

current_num_classes = np.max(y_train) + 1   # this assumes that all classes are present in y


x_train = np.transpose(x_train)
x_test = np.transpose(x_test)

## STEPS 2-3 ---------------------------------------------------------------------
num_to_augment = 50
new_image_filename = './new_img/Aaron_Sorkin/Aaron_Sorkin_0001.jpg'
[aug_images, new_features] = augment_and_extract_features(new_image_filename, num_to_augment)
new_features = np.transpose(new_features)

## STEP 4 ---------------------------------------------------------------------

num_train = 30  # how many out of the augmented new images will be put in train
                # rest will be put in test

new_y_train_cats = np.zeros([1,num_train])
new_y_train_cats[0,:] = current_num_classes
new_y_test_cats = np.zeros([1,num_to_augment-num_train])
new_y_test_cats[0,:] = current_num_classes


y_train = np.concatenate([y_train,new_y_train_cats],1)
y_test = np.concatenate([y_test,new_y_test_cats],1)

y_train = to_categorical(y_train, current_num_classes+1)
y_test = to_categorical(y_test, current_num_classes+1)

y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)
                
current_num_train_samples = y_train.shape[0]
current_num_test_samples = y_test.shape[0]

x_train_aug = np.zeros([x_train.shape[0]+num_train,128])
x_test_aug = np.zeros([x_test.shape[0]+num_to_augment-num_train,128])
x_train_aug[0:x_train.shape[0],:] = x_train
x_test_aug[0:x_test.shape[0],:] = x_test
x_train_aug[x_train.shape[0]:x_train_aug.shape[0],:] = new_features[0:num_train,:]
x_test_aug[x_test.shape[0]:x_test_aug.shape[0],:] = new_features[num_train:num_to_augment,:]

# Add the new samples to the train and test samples
x_train = x_train_aug
x_test = x_test_aug


#%%
## STEP 5 ---------------------------------------------------------------------
## Re-load the trained keras model into temporary temp_model

temp_model = load_model('dlib_classifierV0_trained_c530.h5')

## STEP 6 ---------------------------------------------------------------------
## Increase the size of the output layer, add more weights to network's 2nd hidden layer 
w = temp_model.get_weights()
w_fc2 = w[4] # get the weights for the second fully connected layer
# append weights for another output category to the second fully connected layer
# initialize as random for the new category, keep the weights of the old ones
#   I'm not sure how to control the standard deviation here, but they are limited to (-.05,.05)
new_weights = (2*random([w_fc2.shape[0], 1])-1)/200
w_fc2 = np.concatenate([w_fc2,new_weights],1)
w[4] = w_fc2 # put this back where it was after messing with it
w_out = w[5] # get the weights for the output layer
new_weight = (2*random()-1)/200
w_out = np.append(w_out,new_weight) # add a new random weight to the output layer
w[5] = w_out # put this back where you found it


# Now make the new model with current_C+1 output classes, and stick the new weights in
new_model = Sequential()
new_model.add(Dense(512, kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'relu', input_dim=128, name = 'input'))
new_model.add(Dense(1024, kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'relu', name = 'dense_1'))
new_model.add(Dropout(.5))
new_model.add(Dense(current_num_classes+1, kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'softmax', name = 'dense_2'))

new_model.get_layer('input').set_weights([w[0],w[1]])
new_model.get_layer('dense_1').set_weights([w[2],w[3]])
new_model.get_layer('dense_2').set_weights([w[4],w[5]])

# Freeze all but the last layer
new_model.get_layer('input').trainable = False
new_model.get_layer('dense_1').trainable = False

## STEP 7 ---------------------------------------------------------------------
## Re-Train the model!!

input_dim = 128
batch_size = 64
epochs = 10

new_model.compile(loss = categorical_crossentropy, optimizer = 'SGD', metrics = ['accuracy'])
history = new_model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

overall_score = new_model.evaluate(x_test, y_test, verbose=0)
print('Overall Test loss:', overall_score[0])
print('Overall Test accuracy:', overall_score[1])
#%%
    # this is used to evaluate how well the new net does on JUST the new images
x_new_test = new_features
y_new_test = np.concatenate([np.squeeze(to_categorical(new_y_train_cats, current_num_classes+1)),np.squeeze(to_categorical(new_y_test_cats, current_num_classes+1))])
new_only_score = new_model.evaluate(x_new_test, y_new_test, verbose=0)
print(' Test loss on just the new images:', new_only_score[0])
print(' Test accuracy on just the new images:', new_only_score[1])


#%%
# Now save the new model and the new training data
# Save file while be appended with the number of classes in the filename
model_save_str = 'dlib_classifierV0_trained_c'+str(current_num_classes+1)+'.h5'
new_model.save(model_save_str)

# Save the training&test data/labels
train_features_dict = {}
train_features_dict['train_features'] = x_train
sio.savemat('dlib_features/train_features_c'+str(current_num_classes+1)+'.mat',train_features_dict)

test_features_dict = {}
test_features_dict['test_features'] = x_test
sio.savemat('dlib_features/train_features_c'+str(current_num_classes+1)+'.mat',test_features_dict)

train_class_dict = {}
train_class_dict['training_class'] = y_train
sio.savemat('Class/training_class_c'+str(current_num_classes+1)+'.mat',train_class_dict)

test_class_dict = {}
test_class_dict['test_class'] = y_test
sio.savemat('Class/test_class_c'+str(current_num_classes+1)+'.mat',test_class_dict)

