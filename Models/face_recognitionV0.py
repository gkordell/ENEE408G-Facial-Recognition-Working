from keras.utils import to_categorical  
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.initializers import *
from keras.activations import *
import scipy.io as sio
import numpy as np
from sklearn import preprocessing

#import input and output variables
#remember to normalize the data

input_shape = (3,200,200)
num_classes = 530
batch_size = 64
epochs = 10

x_train_raw_1 = sio.loadmat('training_data_1.mat')['training_data1']
x_train_raw_2 = sio.loadmat('training_data_2.mat')['training_data2']
x_train_raw_3 = sio.loadmat('training_data_3.mat')['training_data3']
x_train_raw = np.concatenate((x_train_raw_1,x_train_raw_2),axis = 3)
x_train_raw = np.concatenate((x_train_raw,x_train_raw_3),axis = 3)
del x_train_raw_1
del x_train_raw_2
del x_train_raw_3
x_test_raw = sio.loadmat('test_data.mat')['test_data']
y_train = sio.loadmat('training_class.mat')['training_class']
y_test = sio.loadmat('test_class.mat')['test_class']

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
'''
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_test = scaler.transform(x_test_raw)
'''
#create model
#possibly iniitailize weights according to paper
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation = 'relu', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(64,64), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(64,64), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(128, kernel_size=(128,128), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(128,128), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(256, kernel_size=(256,256), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(256, kernel_size=(256,256), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(256,256), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(512, kernel_size=(512,512), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(512, kernel_size=(512,512), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(512,512), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(512, kernel_size=(512,512), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Conv2D(512, kernel_size=(512,512), strides=(1,1), padding='same', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Dropout(.5))
model.add(Dense(4096, kernel_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes, kernal_initializer = RandomNormal(mean=0.0, stddev=0.01), activation = 'softmax'))

model.save('facial_recognitionV0_untrained.h5')

#compile, fit, and evaluate model
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr = .001, momentum = .9, decay = .0005), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1) #reomved validation_data=(x_test,y_test)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('facial_recognitionV0_trained.h5')

'''
#used convolutions as fully connected FC layers
model.add(Conv2D(4096, kernel_size=(512,512), strides=(1,1), activation = 'relu'))
model.add(Dropout(.5))
model.add(Conv2D(4096, kernel_size=(4096,4096), strides=(1,1), activation = 'relu'))
model.add(Dropout(.5))
model.add(Conv2D(2622, kernel_size=(4096,4096), strides=(1,1), activation = 'softmax'))
'''
