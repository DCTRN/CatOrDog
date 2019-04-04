# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:19:58 2018

@author: Michal
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras import backend as K

img_width, img_height = 150, 150

train_data_dir = 'dataset/training_set'
test_data_dir = 'dataset/test_set'

samples_nb = 8000
test_samples_nb = 2000
batch_size = 32
epochs = 1

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width,img_height, 3)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.load_weights('./second_try.h5')


import numpy as np
from  keras.preprocessing import image


#print("oto wymiary obrazu:")
#print(input_shape)

img = image.load_img("FitDataSet/sikora.jpg", target_size=(150,150))
#print("Oto obraz:")
#print(image)

img_arr = image.img_to_array(img)
#print("oto obraz jako tablica:")
#print(img_arr)

img_arr_scaled = img_arr * (1./255)
#print("Oto przeskalowany obraz:")
#print(img_arr_scaled)
#print("Oto wymiary przeskalowanego obrazu:")
#print(img_arr_scaled.shape)

img_expanded_dims = np.expand_dims(img_arr_scaled, axis=0)
#print("oto tablica bo expand dims:")
#print(img_expanded_dims)
#print("Oto wymiary obrazu po expand dims:")
#print(img_expanded_dims.shape)

result = model.predict(img_expanded_dims)
print("Oto Wynik: ")
print(result)

