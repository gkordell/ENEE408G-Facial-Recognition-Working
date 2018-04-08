# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:06:28 2018

@author: pydea
"""

# This is adapted from https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16
#Load the VGG model
image_size = 112
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
 
# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers:
    layer.trainable = False

## Check the trainable status of the individual layers
#for layer in vgg_conv.layers:
#    print(layer, layer.trainable)
    
train_dir = './train_base_online'
validation_dir = './validation_base'

# Create the Dense-Layers Classifier to go on top of it
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_base_datagen = ImageDataGenerator(
     rescale=1./255,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True,
     fill_mode='nearest')

validation_base_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 40
val_batchsize = 40

train_base_generator = train_base_datagen.flow_from_directory(
       train_dir,
       target_size=(image_size, image_size),
       batch_size=train_batchsize,
       class_mode='categorical')

validation_base_generator = validation_base_datagen.flow_from_directory(
       validation_dir,
       target_size=(image_size, image_size),
       batch_size=val_batchsize,
       class_mode='categorical',
       shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
# Train the model
history = model.fit_generator(
     train_base_generator,
     steps_per_epoch=train_base_generator.samples/train_base_generator.batch_size ,
     epochs=10,
     validation_data=validation_base_generator,
     validation_steps=validation_base_generator.samples/validation_base_generator.batch_size,
     verbose=1)

# Save the model, trained on the pumpkin and watermelon-only dataset
# model.save('small_last4.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


### Now train using online learning

train_dir_new = './train_new_online'
validation_dir_new = './validation_base'

train_new_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_new_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize_new = 1
val_batchsize_new = 40
 
train_new_generator = train_new_datagen.flow_from_directory(
        train_dir_new,
        target_size=(image_size, image_size),
        batch_size=train_batchsize_new,
        class_mode='categorical')
 
validation_new_generator = validation_new_datagen.flow_from_directory(
        validation_dir_new,
        target_size=(image_size, image_size),
        batch_size=val_batchsize_new,
        class_mode='categorical',
        shuffle=False)

# now re-train, with additional samples
history = model.fit_generator(
      train_new_generator,
      steps_per_epoch=train_new_generator.samples/train_new_generator.batch_size ,
      epochs=10,
      validation_data=validation_new_generator,
      validation_steps=validation_new_generator.samples/validation_new_generator.batch_size,
      verbose=1)


## EVERYTHING PAST HERE IS JUST SHOWING PICTURES OF STUFF

# Create a generator for prediction
validation_generator = validation_new_datagen.flow_from_directory(
        validation_dir_new,
        target_size=(image_size, image_size),
        batch_size=validation_new_generator.batch_size,
        class_mode='categorical',
        shuffle=False)
 
# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))
 
# Show the errors
#for i in range(len(errors)):
#    pred_class = np.argmax(predictions[errors[i]])
#    pred_label = idx2label[pred_class]
#     
#    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#        fnames[errors[i]].split('/')[0],
#        pred_label,
#        predictions[errors[i]][pred_class])
#     
#    original = mpimg.imread('{}/{}'.format(validation_dir,fnames[errors[i]]))
#    plt.figure(figsize=[7,7])
#    plt.axis('off')
#    plt.title(title)
#    plt.imshow(original)
#    plt.show()




