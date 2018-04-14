from keras.utils import to_categorical  
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import *
from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np

#define paramters of model
input_dim = 128
num_classes = 530
batch_size = 64
epochs = 35

#import data
x_train = sio.loadmat('DataWithMeanDlib/train_features.mat')['train_features']
x_test = sio.loadmat('DataWithMeanDlib/test_features.mat')['test_features']
y_train = sio.loadmat('Class/training_class.mat')['training_class']
y_test = sio.loadmat('Class/test_class.mat')['test_class']
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#define model
model = Sequential()
model.add(Dense(512, kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'relu', input_dim=input_dim))
model.add(Dense(1024,kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes, kernel_initializer = RandomNormal(mean=0.0,stddev=.01), activation = 'softmax'))

#compile, fit and evaluate model
model.compile(loss = categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
#model.compile(loss = categorical_crossentropy, optimizer = SGD(lr = .01, momentum = .9), metrics = ['accuracy'])

model.save('dlib_classifierV0_untrained.h5')

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('dlib_classifierV0_trained.h5')

        
