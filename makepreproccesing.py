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


def pull_random_pixels(samples_per_class, pixels_per_sample, char_path_train, leaf):
    '''

    :param samples_per_class:
    :param pixels_per_sample:
    :param char_path_train:
    :param leaf:
    :return:
    '''
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


def Make_prepoccessing(modelse, char_path_train, leaf):
    '''

    :param modelse:
    :param char_path_train:
    :param leaf:
    :return:
    '''
    random_pixels = pull_random_pixels(10, 50, char_path_train, leaf)
    plt.figure()
    plt.suptitle('Random Samples From Each Class', fontsize=14, horizontalalignment='center')
    plt.imshow(random_pixels)
    plt.savefig('graph/ramdom_sample_' + str(modelse) + '.png')
    plt.show()

    r, g, b = cv.split(random_pixels)
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.view_init(20, 120)

    pixel_colors = random_pixels.reshape((np.shape(random_pixels)[0] * np.shape(random_pixels)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

    hsv_img = cv.cvtColor(np.uint8(random_pixels), cv.COLOR_RGB2HSV)

    h, s, v = cv.split(hsv_img)
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.view_init(50, 240)

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

    hsv_img = cv.cvtColor(np.uint8(random_pixels), cv.COLOR_RGB2HSV)

    h, s, v = cv.split(hsv_img)
    fig = plt.figure(figsize=(6, 6))
    axis = fig.add_subplot(1, 1, 1)

    axis.scatter(h.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    plt.show()

    lower_bound = (24, 50, 0)
    upper_bound = (55, 255, 255)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Random Pre-Processed Image From Each Class', fontsize=14, y=.92, horizontalalignment='center',
                 weight='bold')

    for i in range(12):
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

    plt.show()


def color_segment_function(img_array):
    '''

    :param img_array:
    :return:
    '''
    img_array = np.rint(img_array)
    img_array = img_array.astype('uint8')
    hsv_img = cv.cvtColor(img_array, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv_img, (24, 50, 0), (55, 255, 255))
    result = cv.bitwise_and(img_array, img_array, mask=mask)
    result = result.astype('float64')
    return result
