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
from math import sqrt, floor
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

char_path_train = './Datasets/train'
char_path_validation = './Datasets/validation'
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
        if count >= 15:
            break

    leaf.append(tableau)
    sample_count.append(tableau1)
    sample_name.append(tableau2)






# https://github.com/Reedr1208/seedling_classification/blob/master/Seedling_Classification.ipynb

def make_graph_count(x, path, y):
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
    plt.savefig('graph/sample_count_' + str(x) + str(modelse) + '.png')
    plt.show()


def make_graph_random_sample():
    fig = plt.figure(figsize=(10, 15))
    fig.suptitle('Random Samples From Each Class', fontsize=14, y=.92, horizontalalignment='center', weight='bold')
    columns = 5
    rows = 12
    for i in range(12):
        sample_class = os.path.join(char_path_train, sample_name[0][i])
        for j in range(1, 6):
            fig.add_subplot(rows, columns, i * 5 + j)
            plt.axis('off')
            if j == 1:
                plt.text(0.0, 0.5, str(sample_name[0][i]).replace(' ', '\n'), fontsize=13, wrap=True)
                continue
            random_image = os.path.join(sample_class, random.choice(os.listdir(sample_class)))
            # from keras.preprocessing.image
            img = image.load_img(random_image, target_size=(150, 150))
            img = image.img_to_array(img)
            img /= 255.
            plt.imshow(img)
    plt.savefig('graph/Random_sample_' + str(modelse) + '.png')
    plt.show()




def start():
    make_list(char_path_train, 'train')
    make_list(char_path_validation, 'validation')
    make_graph_count('train', char_path_train, 0)
    make_graph_count('validation', char_path_validation, 1)
    make_graph_random_sample()

start()