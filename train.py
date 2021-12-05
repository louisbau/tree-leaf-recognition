import os
import numpy as np
import caer
import tensorflow as tf
from keras.preprocessing import image
import random
import shutil
import makegraph
import makepreproccesing
from prettytable import PrettyTable
import pandas as pd

char_path_train = './Datasets/train'
char_path_validation = './Datasets/validation'
char_path_test = './Datasets/test'
model_version = 24
modelse = 'model_' + str(model_version)

IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32
EPOCHS = 50
dict = {}
leaf = []
sample_count = []
sample_name = []
class_weight = {}
label_map = {}


# TODO  : create_validation(), weigth(), main 2 partie,
# VALIDE : make_list, create_model, train, main 1 partie

def make_list(path, x):
    """
    cette fonction remplie 3 tableau (leaf, sample_count, sample_name) qui vont facilité l'accès au image et à leur tri
    :param path:
    :param x:
    :return:
    """
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


def create_validation(validation_split):
    """
    cette fonction créer un dossier avec 20% des feuilles qui se trouve dans le dossier TRAIN
    :param validation_split:
    :return:
    """
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
    """
    cette fonction défini le model que nous allons utilisé (2 layers)
    :return:
    """
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Dropout(0.5))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    cnn.add(tf.keras.layers.Dense(units=len(leaf[0]), activation='softmax'))
    cnn.summary()

    return cnn


def weigth(training_set):
    """
    Fonction qui calcule le poid de chacune des classes
    :param training_set:
    :return:
    """
    for k, v in training_set.class_indices.items():
        label_map[v] = k

    class_counts = pd.Series(training_set.classes).value_counts()

    for i, c in class_counts.items():
        class_weight[i] = 1.0 / c

    norm_factor = np.mean(list(class_weight.values()))

    for k in class_counts.keys():
        class_weight[k] = class_weight[k] / norm_factor

    t = PrettyTable(['class_index', 'class_label', 'class_weight'])
    for i in sorted(class_weight.keys()):
        t.add_row([i, label_map[i], '{:.2f}'.format(class_weight[i])])
    print(t)


def train(model, x, background):
    """
    Fonction a définir TODO
    :param model:
    :return:
    """
    if background:
        train_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=makepreproccesing.color_segment_function,
            fill_mode='nearest')
        test_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            preprocessing_function=makepreproccesing.color_segment_function,
            fill_mode='nearest')
    else:
        train_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')
        test_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
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

    weigth(training_set)
    if not x:
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        History = model.fit(training_set, validation_data=test_set, epochs=EPOCHS, class_weight=class_weight)

        makegraph.make_graph_accuracy(History, modelse)
        makegraph.make_graph_loss(History, modelse)
        if background:
            model.save_weights('model/background/' + str(modelse) + '.h5')
        else:
            model.save_weights('model/normal/' + str(modelse) + '.h5')
        print('le model a été sauvegarder comme étant ' + str(modelse) + '.h5')


def main():
    """
    Fonction principale qui séquence le programme
    1. Préparation des données
    2. La visualisation des données à l'aide de graph
    3. Creer un nouveau model
    4  charge le model si il a été train || Sinon le train s'effectue et affiche sont efficacité
    5. lance la prediction et affiche les resultat
    :return:
    """
    background = False
    if not os.path.exists('./model'):
        os.mkdir('./model')
    if not os.path.exists('./model/background'):
        os.mkdir('./model/background')
    if not os.path.exists('./model/normal'):
        os.mkdir('./model/normal')
    if not os.path.exists('./graph/' + str(modelse)):
        os.mkdir('./graph/' + str(modelse))

    # Préparation des données
    create_validation(0.2)
    make_list(char_path_train, 'train')
    make_list(char_path_validation, 'validation')

    # Création des graph
    makegraph.make_graph_count('train', 0, leaf, sample_count, sample_name, modelse)
    makegraph.make_graph_count('validation', 1, leaf, sample_count, sample_name, modelse)
    makegraph.make_graph_random_sample(char_path_train, sample_name, modelse)

    # Création du préprocessing
    makepreproccesing.make_prepoccessing(modelse, char_path_train, leaf)
    if input("Voulez vous utilisé la méthode avec background (pas optimisé) y ou n : ") == 'y':
        background = True
    # Création du model
    model = create_model()
    if background:
        if os.path.exists('model/background/' + str(modelse) + '.h5'):
            model.load_weights('model/background/' + str(modelse) + '.h5')
            train(model, True, background)
        else:
            train(model, False, background)
    else:
        if os.path.exists('model/normal/' + str(modelse) + '.h5'):
            model.load_weights('model/normal/' + str(modelse) + '.h5')
            train(model, True, background)
        else:
            train(model, False, background)

    if background:
        test_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            preprocessing_function=makepreproccesing.color_segment_function,
            fill_mode='nearest')
    else:
        test_datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            # preprocessing_function=makepreproccesing.color_segment_function,
            fill_mode='nearest')

    test_generator = test_datagen.flow_from_directory(
        char_path_test,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    # PREDICTION
    print('---- Résultat de la prédiction -----')
    result = model.predict(test_generator, steps=test_generator.n, verbose=1)
    predicted_class_indices = np.argmax(result, axis=1)
    prediction_labels = [label_map[k] for k in predicted_class_indices]
    filenames = test_generator.filenames
    headers = ['file', 'species']
    t = PrettyTable(headers)
    for i, f, p in zip(range(len(filenames)), filenames, prediction_labels):
        if i < 10:
            t.add_row([os.path.basename(f), p])
        elif i < 13:
            t.add_row(['.', '.'])
    print(t)


if __name__ == '__main__':
    main()
