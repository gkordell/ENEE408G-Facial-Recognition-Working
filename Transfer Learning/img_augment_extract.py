# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:49:30 2018

@author: pydea
"""

## AUGMENT THE INPUT IMAGE ---------------------------------------------------------------------
# Change the batchsize according to your system RAM
batchsize = 64

train_datagen_obj = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
test_datagen_obj = ImageDataGenerator(rescale=1./255)

 
train_generator = train_datagen_obj.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')
 
test_generator = test_datagen_obj.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)