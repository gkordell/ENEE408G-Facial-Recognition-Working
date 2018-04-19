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

#might want to run this script for augmented input images as well
batch_size = 2

if __name__ == "__main__":
    model = load_model('dlib_classifierV0_trained.h5')
    input = np.fromstring(sys.argv[1],dtype=float,sep=';')
    input = np.reshape(input,(2,128))
    pred = model.predict(input, batch_size = batch_size)
    sio.savemat('predict.mat',{'pred':pred})
    
