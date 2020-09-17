# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 03:46:19 2020

@author: Nafis
"""
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('asl_alphabet_train/asl_alphabet_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('asl_alphabet_test/asl_alphabet_test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size = (4, 4)))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 26, activation = 'softmax'))
classifier.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(training_set, nb_epoch = 100, validation_data = test_set)

classifier.summary()
classifier.save('f_m_CNN.h5')


