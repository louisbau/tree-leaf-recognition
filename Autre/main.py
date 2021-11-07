import cv2 as cv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caer
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing import image

char_path = '../Datasets/train'
char_dict = {}
IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 30

for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

char_dict = caer.sort_dict(char_dict, descending=True)
print(len(char_dict))
# training image proccessing

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('./Datasets/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                 class_mode='categorical')

# test image
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('./Datasets/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                            class_mode='categorical')

# building model

cnn = tf.keras.models.Sequential()

# building convolution layer

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=len(char_dict), activation='softmax'))

cnn.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

cnn.fit(x = training_set , validation_data = test_set , epochs = EPOCHS)


test_image = image.load_img('../Datasets/test/diospyros_virginiana/12992000042158.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
print(result)
