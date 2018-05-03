#args are dlib vectors saves predictions in a .mat file
import sys
from keras.utils import to_categorical  
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import *
from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np
from img_augment_extract import augment_and_extract_features
from scipy import stats

#might want to run this script for augmented input images as well
batch_size = 2

if __name__ == "__main__":
    filename = sys.argv[1]
    #filename = 'testfile.jpg'
    current_num_classes = int(sys.argv[2])
    if current_num_classes == 530:
        model = load_model('./keras_models/dlib_classifierV0_trained.h5')
        x_train = sio.loadmat('dlib_features/train_features.mat')['train_features']
    else:
        model = load_model('./keras_models/dlib_classifierV0_trained_modified.h5')
        x_train = sio.loadmat('dlib_features/train_features_modified.mat')['train_features']
    
    features = augment_and_extract_features(filename,20)
    #features = np.transpose(features)
    cats = np.zeros((1,20))
    for i in range(20):
        cats[0,i] = 532
    cats = np.squeeze(cats)
    cats = to_categorical(cats,533)
    #input = np.fromstring(sys.argv[1],dtype=float,sep=';')
    #input = np.reshape(input,(2,128))
    #result = model.evaluate(features, cats, batch_size = batch_size)
    predictions = model.predict(features, batch_size = batch_size)
    ids = np.argmax(predictions,1)
    identity = stats.mode(ids)
    confidence = np.mean(predictions[:,identity.mode[0]])
    minconfidence = np.min(predictions[:,identity.mode[0]])
    
    min_dist = 100000
    for j in range(0,np.shape(x_train)[0],3): 
        dist = 0
        for k in range(128):
            dist = dist + (x_train[j,k] - features[0,k])**2
        if dist < min_dist:
            #min_distances_mod[int(i/10)] = dist
            min_dist= dist
        
    if min_dist > 0.4:    #is the minimum distance to the dataset is above 1, this is a false positive
        confidence = 0
    
    sio.savemat('pred.mat',{'pred':identity.mode[0], 'confidence':confidence, 'min_dist':min_dist,'num':current_num_classes})
    #sio.savemat('predict.mat',{'pred':pred})
    
