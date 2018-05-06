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
from img_augment_extract import augment_and_extract_features
# from keras.preprocessing.image import ImageDataGenerator
# import dlib

if __name__ == "__main__":
	#define paramters of model
    aug_num = 50
    epochs = 10
    
	### ONLINE LEARNING ---------------------------------------------------------
	## STEP 1 ---------------------------------------------------------------------
	#import new image, get features
    x_train = augment_and_extract_features(sys.argv[1], aug_num)
   #x_train = augment_and_extract_features('testfile.jpg', aug_num)
    y_train = np.ones([1, aug_num])*int(sys.argv[2])
	# get testing data
    current_num_classes = int(sys.argv[3])
#    x_train = augment_and_extract_features('testfile.jpg', aug_num)
#    y_train = np.ones([1, aug_num])*int(530)
#    	# get testing data
#    current_num_classes = int(531)
    if current_num_classes == 530:
        x_test = sio.loadmat('./dlib_features/test_features.mat')['test_features']
        y_test = sio.loadmat('./Class/test_class.mat')['test_class']
    else:
        x_test = sio.loadmat('./dlib_features/test_features_modified.mat')['test_features']
        y_test = sio.loadmat('./Class/test_class_modified.mat')['test_class'] 
    
    
    current_num_classes = np.max(y_test) + 1   # this assumes that all classes are present in y
    
    if x_train.shape[0] == 128: # then they need to be transposed
        x_train = np.transpose(x_train)
        x_test = np.transpose(x_test)
    
    y_train = to_categorical(y_train, current_num_classes)
    y_test = to_categorical(y_test, current_num_classes)
    
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    
    	## STEP 2 ---------------------------------------------------------------------
    	#load pretrained model
    if current_num_classes == 530:
        model = load_model('./keras_models/dlib_classifierV0_trained.h5')
    else:
        model = load_model('./keras_models/dlib_classifierV0_trained_modified.h5')
    
    	## STEP 3 ---------------------------------------------------------------------
    	#retrain model with new face
    
    history = model.fit(
            x_train, 
            y_train,
            epochs = epochs,
            validation_data = (x_test, y_test))
    
    
    model.save_weights('./keras_models/dlib_classifierV0_trained_modified.h5')


#train_acc = history.history['acc']
#val_acc = history.history['val_acc']
#train_loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#x = range(len(train_acc))
#import matplotlib.pyplot as plt
#plt.plot(x, train_acc, 'b', label='Training Accuracy on New Image')
#plt.plot(x, val_acc, 'r', label='Validation Accuracy on Whole Dataset')
#plt.title('Training and Validation accuracy')
#plt.legend()
# 
#plt.figure()