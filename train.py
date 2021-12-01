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
import makepreproccesing
from keras import models, layers, callbacks
from prettytable import PrettyTable
import pandas as pd

char_path_train = './Datasets/train'
char_path_validation = './Datasets/validation'
char_path_test = './Datasets/test'
modelse = 'model_18'

IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 50
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
        if count >= 10:
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
    # train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # test image
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        # preprocessing_function=makepreproccesing.color_segment_function,
        fill_mode='nearest')
    test_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        # preprocessing_function=makepreproccesing.color_segment_function,
        fill_mode='nearest')

    training_set = train_datagen.flow_from_directory(
        char_path_train,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=leaf[0])
    test_set = test_datagen.flow_from_directory(
        char_path_validation,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical', classes=leaf[0])
    test_generator = test_datagen.flow_from_directory(
        char_path_test,
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)
    label_map = {}
    for k, v in training_set.class_indices.items():
        label_map[v] = k

    class_counts = pd.Series(training_set.classes).value_counts()
    class_weight = {}

    for i, c in class_counts.items():
        class_weight[i] = 1.0 / c

    norm_factor = np.mean(list(class_weight.values()))

    for k in class_counts.keys():
        class_weight[k] = class_weight[k] / norm_factor

    t = PrettyTable(['class_index', 'class_label', 'class_weight'])
    for i in sorted(class_weight.keys()):
        t.add_row([i, label_map[i], '{:.2f}'.format(class_weight[i])])
    print(t)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    h1 = 'model/' + str(modelse) + '.h5'
    '''
    best_cb = callbacks.ModelCheckpoint(h1,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        save_freq='epoch',
                                        mode='auto')
    '''



    History = model.fit(training_set, validation_data=test_set, epochs=EPOCHS, class_weight=class_weight)
                        #callbacks=[best_cb]
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_accuracy_' + str(modelse) + '.png')
    plt.show()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_loss_' + str(modelse) + '.png')
    plt.show()
    #model.load_weights('model_best.h5')
    model.save_weights('model/' + str(modelse) + '.h5')
    print('le model a été sauvegarder comme étant ' + str(modelse) + '.h5')


def main():
    create_validation(0.2)
    make_list(char_path_train, 'train')
    make_list(char_path_validation, 'validation')
    model = create_model()


    if os.path.exists('model/' + str(modelse) + '.h5'):
        model.load_weights('model/' + str(modelse) + '.h5')
    else:
        train(model)


    # test image
    test_image = image.load_img('Datasets/test/feuille_test/abies_concolor.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    # print(training_set.class_indices)
    for i, name in enumerate(leaf[0]):
        print(i, ' : ', name)

    print(result)
    print(leaf[0])
    for i in range(len(result[0])):
        if result[0][i] != 0:
            print(i)
            print("resultat logique : " + str(leaf[0][i] + ', d\'indice ' + str(i)))

main()
