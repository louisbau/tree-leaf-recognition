import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import random


# https://github.com/Reedr1208/seedling_classification/blob/master/Seedling_Classification.ipynb

def make_graph_count(x, y, leaf, sample_count, sample_name, modelse):
    """
    fonction qui affiche le nombre de sample par classe
    :param x:
    :param y:
    :param leaf:
    :param sample_count:
    :param sample_name:
    :param modelse:
    :return:
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    y_pos = np.arange(len(leaf[y]))

    ax.barh(y_pos, sample_count[y], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sample_name[y])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Sample Counts')
    ax.set_title(str(x) + ' Sample Counts Per Class')
    plt.savefig('graph/' + str(modelse) + '/sample_count_' + str(x) + str(modelse) + '.png')
    plt.show()


def make_graph_accuracy(History, modelse):
    """
    fonction qui affiche la précision du training
    :param History:
    :param modelse:
    :return:
    """
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/' + str(modelse) + '/model_accuracy_' + str(modelse) + '.png')
    plt.show()


def make_graph_loss(History, modelse):
    """
    fonction qui affiche les pertes du training
    :param History:
    :param modelse:
    :return:
    """
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/' + str(modelse) + '/model_loss_' + str(modelse) + '.png')
    plt.show()


def make_graph_random_sample(char_path_train, sample_name, modelse):
    """
    fonction qui affiche différentes feuilles associées a une classe
    :param char_path_train:
    :param sample_name:
    :param modelse:
    :return:
    """
    fig = plt.figure(figsize=(10, 15))
    fig.suptitle('Random Samples From Each Class', fontsize=14, y=.92, horizontalalignment='center', weight='bold')
    columns = 5
    rows = 10
    for i in range(10):
        sample_class = os.path.join(char_path_train, sample_name[0][i])
        for j in range(1, 6):
            fig.add_subplot(rows, columns, i * 5 + j)
            plt.axis('off')
            if j == 1:
                plt.text(0.0, 0.5, str(sample_name[0][i]).replace(' ', '\n'), fontsize=13, wrap=True)
                continue
            random_image = os.path.join(sample_class, random.choice(os.listdir(sample_class)))
            img = image.load_img(random_image, target_size=(150, 150))
            img = image.img_to_array(img)
            img /= 255.
            plt.imshow(img)
    plt.savefig('graph/' + str(modelse) + '/Random_sample_' + str(modelse) + '.png')
    plt.show()
