import cv2 as cv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caer
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing import image
import random
import shutil
import makegraph

char_path_train = './Datasets/train'
char_path_validation = './Datasets/validation'
char_path_test = './Datasets/test'
models = 'model_02'

IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 30
dict = {}
leaf = []
sample_count = []
sample_name = []


def make_list(path, x):
    dicts = {}
    for char in os.listdir(path):
        dicts[char] = len(os.listdir(os.path.join(path, char)))
    dicts = caer.sort_dict(dicts, descending=True)
    dict[x] = dicts
    count = 0
    tableau = []
    tableau1 = []
    tableau2 = []
    for i in dict[x]:
        tableau.append(i[0])
        tableau1.append(i[1])
        tableau2.append(i[0])
        count += 1
        if count >= 15:
            break

    leaf.append(tableau)
    sample_count.append(tableau1)
    sample_name.append(tableau2)


# https://github.com/Reedr1208/seedling_classification/blob/master/Seedling_Classification.ipynb


def create_validation(validation_split):
    if os.path.isdir(char_path_validation):
        print('Validation directory already created!')
        print('Process Terminated')
        return
    os.mkdir(char_path_validation)
    for f in os.listdir(char_path_train):
        train_class_path = os.path.join(char_path_train, f)
        if os.path.isdir(train_class_path):
            validation_class_path = os.path.join(char_path_validation, f)
            os.mkdir(validation_class_path)
            files_to_move = int(validation_split * len(os.listdir(train_class_path)))

            for i in range(files_to_move):
                random_image = os.path.join(train_class_path, random.choice(os.listdir(train_class_path)))
                shutil.move(random_image, validation_class_path)
    print('le folder de validation représente {:.2%} du folder train'.format(validation_split))


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

    cnn.add(tf.keras.layers.Dense(units=len(leaf[0]), activation='softmax'))

    return cnn


def train(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(char_path_train, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf[0])

    # test image
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(char_path_validation, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                class_mode='categorical', classes=leaf[0])
    makegraph.make_graph_accuracy(model, training_set, test_set)
    '''
    History = model.fit(x=training_set, validation_data=test_set, epochs=EPOCHS)

    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_accuracy_' + str(models) + '.png')
    plt.show()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_loss_' + str(models) + '.png')
    plt.show()
    '''
    model.save_weights('model/' + str(models) + '.h5')
    print('le model a été sauvegarder comme étant ' + str(models) + '.h5')


def main():
    create_validation(0.2)
    make_list(char_path_train, 'train')
    make_list(char_path_validation, 'validation')
    model = create_model()
    if os.path.exists('model/' + str(models) + '.h5'):
        model.load_weights('model/' + str(models) + '.h5')
    else:
        train(model)
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(char_path_train, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf[0])

    # test image
    test_image = image.load_img('Datasets/test/acer_rubrum/pi2608-04-2.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    print(training_set.class_indices)
    print(result)
    print(leaf[0])
    for i in range(len(result[0])):
        if result[0][i] == 1:
            print(i)
            print("resultat logique : " + str(leaf[0][i] + ', d\'indice ' + str(i)))


main()
