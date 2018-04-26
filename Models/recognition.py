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
    model = load_model('dlib_classifierV0_trained_modified.h5')
    
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
    print(identity)
    #sio.savemat('predict.mat',{'pred':pred})
    
