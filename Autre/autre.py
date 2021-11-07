import cv2 as cv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import caer
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

char_path = '../Datasets/train'
char_dict = {}
IMG_SIZE = (64, 64)
channels = 1
BATCH_SIZE = 32

for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

char_dict = caer.sort_dict(char_dict, descending=True)
print(char_dict)

leaf = []
count = 0
for i in char_dict:
    leaf.append(i[0])
    count += 1
    if count >= 10:
        break

print(len(char_dict))
print(leaf)