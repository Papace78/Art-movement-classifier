import pandas as pd
import numpy as np

import matplotlib as plt
import seaborn as sns

import os
import random
import glob
import cv2

import tensorflow as tf

PATH = DATASET_PATH

def get_data(path, batch_size=None, as_iterator=False):
    data =  tf.keras.utils.image_dataset_from_directory(path, batch_size=batch_size)
    data = data.map(lambda x, y: (x/255, y))

    if as_iterator:
        data = data.as_numpy_iterator()

    return data

def get_class_data(style):
    images = []

    for img in os.listdir(os.path.join(PATH, style)):
        img_path = os.path.join(PATH, style, img)
        images.append(cv2.imread(img_path))

    return images


def plot_paintings(nb, styles: list):

    for style in styles:
        fig, axs = plt.subplots(nrows=1, ncols=nb, figsize = (15,3))
        random_list = random.choices(os.listdir(os.path.join(PATH,style)), k = nb)
        fig.suptitle(style)

        for j, img in enumerate(random_list):
            image = plt.imread(os.path.join(PATH,style,img))
            axs[j].axis("off")
            axs[j].imshow(image)

    plt.show()
