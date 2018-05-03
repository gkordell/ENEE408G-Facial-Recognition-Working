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
from keras.callbacks import Callback
from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np
from numpy.random import random
from keras.preprocessing.image import ImageDataGenerator
import dlib
from os import listdir
import sys

from img_augment_extract import augment_and_extract_features
if __name__ == "__main__":
    new_img_dir = sys.argv[1]
    current_num_classes = int(sys.argv[2])
    #new_img_dir = './newface'
    ## STEP 1 ---------------------------------------------------------------------
    #import data
    if current_num_classes != 530:
        x_train = sio.loadmat('dlib_features/train_features_modified.mat')['train_features']
        x_test = sio.loadmat('dlib_features/test_features_modified.mat')['test_features']
        y_train = sio.loadmat('Class/training_class_modified.mat')['training_class']
        y_test = sio.loadmat('Class/test_class_modified.mat')['test_class']
    else:   # if this is the first run of transfer learning, use the original versions
        x_train = sio.loadmat('dlib_features/train_features.mat')['train_features']
        x_test = sio.loadmat('dlib_features/test_features.mat')['test_features']
        y_train = sio.loadmat('Class/training_class.mat')['training_class']
        y_test = sio.loadmat('Class/test_class.mat')['test_class']
    
    #current_num_classes = int(np.max(y_train) + 1)   # this assumes that all classes are present in y
    
    if x_train.shape[1] > x_train.shape[0]: # then they need to be transposed
        x_train = np.transpose(x_train)
        x_test = np.transpose(x_test)
    
    ## STEPS 2-3 ---------------------------------------------------------------------
    num_to_generate = 140
    #new_image_dir = './new_img/Gordon_Kordell/'
    fnames = listdir(new_img_dir)
    new_features = np.zeros((1,128))
    num_to_augment = int(num_to_generate/len(fnames) + 1)
    for name in fnames:
        temp_features = augment_and_extract_features(new_img_dir+'/'+name, num_to_augment)
        #temp_features = np.transpose(temp_features)
        new_features = np.concatenate([new_features,temp_features])
    new_features = new_features[1:-1,:]
    num_generated = new_features.shape[0]
    
    ## STEP 4 ---------------------------------------------------------------------
    
    num_train = 140   # how many out of the augmented new images will be put in train
                    # rest will be put in test
    
    new_y_train_cats = np.zeros([1,num_train])
    new_y_train_cats[0,:] = current_num_classes
    new_y_test_cats = np.zeros([1,num_generated-num_train])
    new_y_test_cats[0,:] = current_num_classes
    
    # repeat each new image in the training set to require fewer epochs to improve accuracy on new images
    # the repeated versions won't be saved in the new modified dataset
    num_to_repeat = 1
    for i in range(num_to_repeat):
        y_train = np.concatenate([y_train,new_y_train_cats],1)
        
    y_test = np.concatenate([y_test,new_y_test_cats],1)
    
    y_train = to_categorical(y_train, current_num_classes+1)
    y_test = to_categorical(y_test, current_num_classes+1)
    
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
                    
    current_num_train_samples = y_train.shape[0]
    current_num_test_samples = y_test.shape[0]
    
    x_train_aug = np.zeros([x_train.shape[0]+num_train,128])
    x_test_aug = np.zeros([x_test.shape[0]+num_generated-num_train,128])
    x_train_aug[0:x_train.shape[0],:] = x_train
    x_test_aug[0:x_test.shape[0],:] = x_test
    
    # repeat each new training sample as above with y_train
    for i in range(num_to_repeat):
        x_train = np.concatenate([x_train,new_features[0:num_train,:]])
    x_test_aug[x_test.shape[0]:x_test_aug.shape[0],:] = new_features[num_train:num_generated,:]
    
    # Add the new samples to the test samples
    x_test = x_test_aug
    
    
    #%%
    ## STEP 5 ---------------------------------------------------------------------
    ## Re-load the trained keras model into temporary temp_model
    if current_num_classes != 530:
        temp_model = load_model('./keras_models/dlib_classifierV0_trained_modified.h5')
    else:
        temp_model = load_model('./keras_models/dlib_classifierV0_trained.h5')
    
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
    #new_model.get_layer('input').trainable = False
    #new_model.get_layer('dense_1').trainable = False
    
    ## STEP 7 ---------------------------------------------------------------------
    ## Re-Train the model!!
    
    input_dim = 128
    batch_size = 64
    
    x_added_test = x_test[15519:-1,:]
    y_added_test = y_test[15519:-1,:]
    x_added_train = x_train[36213:-1,:]
    y_added_train = y_train[36213:-1,:]
    
    new_img_acc_hist = []
    
    # this is used to evaluate how well the new net does on JUST the added images
    # it gets called on the end of each epoch as a callback in the fit function
    class NewImgAccuracy(Callback):
        def on_epoch_end(self, batch, logs={}):
            new_only_score = new_model.evaluate(x_added_test, y_added_test, verbose=0)
            print(' Test loss on just the new images:', new_only_score[0])
            print(' Test accuracy on just the new images:', new_only_score[1])
            new_img_acc_hist.append(new_only_score[1])
    
    print_new_acc = NewImgAccuracy()
    
    new_model.compile(loss = categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
    epochs = 5
    history = new_model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [print_new_acc])
    
    overall_score = new_model.evaluate(x_test, y_test, verbose=0)
    print('Overall Test loss:', overall_score[0])
    print('Overall Test accuracy:', overall_score[1])

    #%%
    # Now save the new model and the new training data
    # Save file while be appended with the number of classes in the filename
    model_save_str = './keras_models/dlib_classifierV0_trained_modified.h5'
    new_model.save(model_save_str)
    
    # convert y-train and y-test back to single vector and not one-hot matrix
    y_train = np.argmax(y_train,1)
    y_test = np.argmax(y_test,1)
    
    # Save the training&test data/labels
    train_features_dict = {}
    train_features_dict['train_features'] = x_train
    sio.savemat('dlib_features/train_features_modified.mat',train_features_dict)
    
    test_features_dict = {}
    test_features_dict['test_features'] = x_test
    sio.savemat('dlib_features/test_features_modified.mat',test_features_dict)
    
    train_class_dict = {}
    train_class_dict['training_class'] = y_train
    sio.savemat('Class/training_class_modified.mat',train_class_dict)
    
    test_class_dict = {}
    test_class_dict['test_class'] = y_test
    sio.savemat('Class/test_class_modified.mat',test_class_dict)
    
#%% Plotting
    
    
#train_acc = history.history['acc']
#val_acc = history.history['val_acc']
#train_loss = history.history['loss']
#val_loss = history.history['val_loss']

#x = range(len(train_acc))
#import matplotlib.pyplot as plt
#plt.plot(x, train_acc, 'b', label='Training Accuracy')
#plt.plot(x, val_acc, 'r', label='Validation Accuracy')
#plt.title('Training and Validation accuracy')
#plt.legend()
 
#plt.figure()
