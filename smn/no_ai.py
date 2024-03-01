import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import skimage.transform



def average_color(image_path, return_table = True):
    image = io.imread(image_path)
    #calculate average per channel
    average_per_channel = np.mean(np.mean(image,axis = 0), axis = 0).astype(int)

    if not return_table:
        return average_per_channel

    #create a ndarray of shape image with only 1 and multiply it by the average value per channel
    average_table = (np.ones(shape = image.shape)*average_per_channel).astype(int)

    return image, average_table #ndarray of original's image shape


def dominant_colors(image_path, return_table = True):

    image = io.imread(image_path)

    #from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    #using cv2 K-Means clustering

    ## INPUT

    #samples = reshaped image so all pixels as index * 3 columns (RGB)
    pixels = np.float32(image.reshape(-1, 3))
    #n_clusters =
    n_colors = 5
    #criteria: iteration termination condition (type, max, epsilon accuracy)=
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    #attemps: number of times algorithm is initialized =
    attempts = 10
    #flags: how initial centers are taken =
    flags = cv2.KMEANS_RANDOM_CENTERS

    ## OUTPUT

    #compactness: sum of squared distance from each point to their centers (oseeef)
    #labels: array labeling from 0 to n_colors each pixel
    #centers: array of centers of clusters (each center is a color)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, attempts, flags)

    #count number of times each label appears
    _, counts = np.unique(labels, return_counts=True)

    if not return_table:
        indices = np.argsort(counts)[::-1]
        color0 = palette[indices[0]]
        color1 = palette[indices[1]]
        color2 = palette[indices[2]]
        color3 = palette[indices[3]]
        color4 = palette[indices[4]]

        return color0,color1,color2,color3,color4

    #delimiate and draw the dominant colors table
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(image.shape[0]*freqs)

    dom_patch = np.zeros(shape=image.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    #resize dom_patch to a square
    dom_patch = skimage.transform.resize(dom_patch,output_shape = (226,226,3))

    return image, dom_patch


def draw_table(table: np.array, image: np.array = False, title: str = 'table'):

    if type(image) == type(np.ones(shape=(1,1))):
        fig, ax = plt.subplots(1, 2, figsize = (10,6))

        ax[0].imshow(image)
        ax[0].axis('off')

        ax[1].imshow(table, extent=[0,100,0,100])
        ax[1].axis('off')
        ax[1].set_title(f'{title}')

        plt.savefig(f'{title}.png')
        plt.close()

    else:
        plt.imshow(table, extent=[0,100,0,100])
        plt.axis('off')
        plt.title(f'{title}')

        plt.savefig(f'{title}.png')
        plt.close()
