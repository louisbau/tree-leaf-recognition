import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt, floor
from matplotlib import colors

"""
GROUPE 3TL1: 
LOUIS BAUCHAU
LOGAN MONTALTO
DEVASHISH BASNET 
BRICE KOUETCHEU
"""


# STEP 1
def pull_random_pixels(samples_per_class, pixels_per_sample, char_path_train, leaf):
    """
    fonction qui récupère aléatoirement des pixels des images sélectionnées
    :param samples_per_class:
    :param pixels_per_sample:
    :param char_path_train:
    :param leaf:
    :return:
    """
    total_pixels = len(leaf[0]) * samples_per_class * pixels_per_sample
    random_pixels = np.zeros((total_pixels, 3), dtype=np.uint8)
    for i in range(len(leaf[0])):
        sample_class = os.path.join(char_path_train, leaf[0][i])
        for j in range(samples_per_class):
            random_image = os.path.join(sample_class, random.choice(os.listdir(sample_class)))
            img = cv.imread(random_image)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = np.reshape(img, (img.shape[0] * img.shape[1], 3))
            new_pixels = img[np.random.randint(0, img.shape[0], pixels_per_sample)]

            start_index = pixels_per_sample * (i * samples_per_class + j)
            random_pixels[start_index:start_index + pixels_per_sample, :] = new_pixels

    h = floor(sqrt(total_pixels))
    w = total_pixels // h

    random_pixels = random_pixels[np.random.choice(total_pixels, h * w, replace=False)]
    random_pixels = np.reshape(random_pixels, (h, w, 3))
    return random_pixels


# STEP 2
def make_graph_collect_random_pixel(random_pixels, modelse):
    """
    fonction qui va afficher la valeur des pixels dans plan RGB du step 1
    :param random_pixels:
    :param modelse:
    :return:
    """
    plt.figure()
    plt.suptitle('Random Samples From Each Class', fontsize=14, horizontalalignment='center')
    plt.imshow(random_pixels)
    plt.savefig('graph/' + str(modelse) + '/collect_random_pixel.png')
    plt.show()


# STEP 3
def make_graph_display_color(random_pixels, modelse, pixel_colors):
    """
    fonction qui va afficher la valeur des pixels dans espace RGB
    :param random_pixels:
    :param modelse:
    :param pixel_colors:
    :return:
    """
    r, g, b = cv.split(random_pixels)
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle('Random Samples From Each Class In A Space RGB', fontsize=14, horizontalalignment='center')
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.view_init(20, 120)

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.savefig('graph/' + str(modelse) + '/display_color_rgb.png')
    plt.show()


# STEP 4
def make_graph_display_hsv_3d(random_pixels, modelse, pixel_colors):
    """
    fonction qui va afficher la teinte, la saturation la valeurs des pixels dans un espace SHV
    :param random_pixels:
    :param modelse:
    :param pixel_colors:
    :return:
    """
    hsv_img = cv.cvtColor(np.uint8(random_pixels), cv.COLOR_RGB2HSV)

    h, s, v = cv.split(hsv_img)
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle('Random Samples From Each Class In A Space HSV', fontsize=14, horizontalalignment='center')
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.view_init(50, 240)

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.savefig('graph/' + str(modelse) + '/display_method_HSV_3D.png')
    plt.show()


# STEP 5
def make_graph_display_hsv_2d(random_pixels, modelse, pixel_colors):
    """
    fonction qui va afficher la teinte et la saturation des pixels dans un plan SHV
    :param random_pixels:
    :param modelse:
    :param pixel_colors:
    :return:
    """
    hsv_img = cv.cvtColor(np.uint8(random_pixels), cv.COLOR_RGB2HSV)

    h, s, v = cv.split(hsv_img)
    fig = plt.figure(figsize=(6, 6))
    plt.suptitle('Random Samples From Each Class In A Plan HSV', fontsize=14, horizontalalignment='center')
    axis = fig.add_subplot(1, 1, 1)

    axis.scatter(h.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    plt.savefig('graph/' + str(modelse) + '/display_method_HSV_2D.png')
    plt.show()


# STEP 6
def make_display_pre_process(char_path_train, leaf, modelse):
    """
    Fonction qui va appliquer les observations lower bound (HUE) et upper bound (SATURATION), et par conséquence isoler le vert.
    Affiche aussi le résultat
    :param char_path_train:
    :param leaf:
    :param modelse:
    :return:
    """
    lower_bound = (27, 50, 0)  # HUE de 27 -> 50
    upper_bound = (50, 255, 255)  # SATURATION de 50 -> 255

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Random Pre-Processed Image From Each Class', fontsize=14, y=.92, horizontalalignment='center',
                 weight='bold')
    for i in range(10):
        sample_class = os.path.join(char_path_train, leaf[0][i])
        random_image = os.path.join(sample_class, random.choice(os.listdir(sample_class)))
        img = cv.imread(random_image)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (150, 150))

        hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        mask = cv.inRange(hsv_img, lower_bound, upper_bound)
        result = cv.bitwise_and(img, img, mask=mask)

        fig.add_subplot(6, 4, i * 2 + 1)
        plt.imshow(img)
        plt.axis('off')

        fig.add_subplot(6, 4, i * 2 + 2)
        plt.imshow(result)
        plt.axis('off')
    plt.savefig('graph/' + str(modelse) + '/display_preproccesing_img.png')
    plt.show()


# STEP 7 final
def color_segment_function(img_array):
    """
    Fonction finale compatible avec ImageDataGenerator de Keras, qui retire le background, qui découle des autres
    étapes des valeurs de lower bound et upper bound
    :param img_array:
    :return:
    """
    img_array = np.rint(img_array)
    img_array = img_array.astype('uint8')
    hsv_img = cv.cvtColor(img_array, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv_img, (27, 50, 0), (50, 255, 255))
    result = cv.bitwise_and(img_array, img_array, mask=mask)
    result = result.astype('float64')
    return result


def make_prepoccessing(modelse, char_path_train, leaf):
    """
    Fonction principale du pre-processing qui est divisé en 6 étapes
    :param modelse:
    :param char_path_train:
    :param leaf:
    :return:
    """
    # STEP 1 :
    random_pixels = pull_random_pixels(10, 50, char_path_train, leaf)
    pixel_colors = random_pixels.reshape((np.shape(random_pixels)[0] * np.shape(random_pixels)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    # STEP 2 :
    make_graph_collect_random_pixel(random_pixels, modelse)
    # STEP 3 :
    make_graph_display_color(random_pixels, modelse, pixel_colors)
    # STEP 4 :
    make_graph_display_hsv_3d(random_pixels, modelse, pixel_colors)
    # STEP 5 :
    make_graph_display_hsv_2d(random_pixels, modelse, pixel_colors)
    # STEP 6 :
    make_display_pre_process(char_path_train, leaf, modelse)
