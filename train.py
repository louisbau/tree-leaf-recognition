import cv2 as cv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caer
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing import image

char_path = './Datasets/train'

IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 30
dict = {}
for char in os.listdir(char_path):
    dict[char] = len(os.listdir(os.path.join(char_path, char)))

dict = caer.sort_dict(dict, descending=True)
leaf = []
count = 0
for i in dict:
    leaf.append(i[0])
    count += 1
    if count >= 15:
        break


def create_model():
    cnn = tf.keras.models.Sequential()

    # building convolution layer

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.5))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    cnn.add(tf.keras.layers.Dense(units=len(leaf), activation='softmax'))

    return cnn


def train(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('./Datasets/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf)

    # test image
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory('./Datasets/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                class_mode='categorical', classes=leaf)
    History = model.fit(x=training_set, validation_data=test_set, epochs=EPOCHS)

    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    model.save_weights('weigthse.h5')
    print('le model a été sauvegarder comme étant modelPrediction.h5')


def main():
    model = create_model()
    if os.path.exists('weigthse.h5'):
        model.load_weights('weigthse.h5')
    else:
        train(model)
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('./Datasets/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf)

    # test image
    test_image = image.load_img('Datasets/test/diospyros_virginiana/pi0196-05-4.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    print(training_set.class_indices)
    print(result)
    print("quoi ???")
    print(leaf)
    for i in range(len(result[0])):
        if result[0][i] == 1:
            print(i)
            ##print("resultat nonlogique : "+str(leaf[(len(leaf)-1)-i]))
            print("resultat logique : "+str(leaf[i]))


main()
