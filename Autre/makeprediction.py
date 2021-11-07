import cv2 as cv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caer
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing import image

char_path = './datasetProjet/train'

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
    if count >=15:
        break


print(leaf)