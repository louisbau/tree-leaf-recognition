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
models = 'model_01'

IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 30
dict = {}
for char in os.listdir(char_path):
    dict[char] = len(os.listdir(os.path.join(char_path, char)))

dict = caer.sort_dict(dict, descending=True)
leaf = []
sample_count = []
sample_name = []
count = 0
for i in dict:
    leaf.append(i[0])
    sample_count.append(i[1])
    sample_name.append(i[0])
    count += 1
    if count >= 15:
        break
"""
classes= []
sample_counts= []
for f in os.listdir('Datasets/train'):
    train_class_path= os.path.join('Datasets/train', f)
    if os.path.isdir(train_class_path):
        classes.append(f)
        sample_counts.append(len(os.listdir(train_class_path)))
"""





plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
y_pos = np.arange(len(leaf))

ax.barh(y_pos, sample_count, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(sample_name)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Sample Counts')
ax.set_title('Sample Counts Per Class')
plt.savefig('graph/sample_count' + str(models) + '.png')
plt.show()


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
    plt.savefig('graph/model_accuracy' + str(models) + '.png')
    plt.show()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_loss' + str(models) + '.png')
    plt.show()

    model.save_weights('model/' + str(models) + '.h5')
    print('le model a été sauvegarder comme étant ' + str(models) + '.h5')


def main():
    model = create_model()
    if os.path.exists('model/' + str(models) + '.h5'):
        model.load_weights('model/' + str(models) + '.h5')
    else:
        train(model)
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('./Datasets/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf)

    # test image
    test_image = image.load_img('Datasets/test/acer_rubrum/pi2608-04-2.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    print(training_set.class_indices)
    print(result)
    print(leaf)
    for i in range(len(result[0])):
        if result[0][i] == 1:
            print(i)
            print("resultat logique : " + str(leaf[i] + ', d\'indice ' + str(i)))


main()
