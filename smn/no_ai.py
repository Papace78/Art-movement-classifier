import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import skimage.transform
from tqdm import tqdm


def average_color(image_path, return_table=True):
    image = io.imread(image_path)
    # calculate average per channel
    average_per_channel = np.mean(np.mean(image, axis=0), axis=0).astype(int)

    if not return_table:
        return average_per_channel

    # create a ndarray of shape image with only 1 and multiply it by the average value per channel
    average_table = (np.ones(shape=image.shape) * average_per_channel).astype(int)

    return image, average_table  # ndarray of original's image shape


def dominant_colors(image_path, return_table=True):

    image = io.imread(image_path)

    # from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    # using cv2 K-Means clustering

    ## INPUT

    # samples = reshaped image so all pixels as index * 3 columns (RGB)
    pixels = np.float32(image.reshape(-1, 3))
    # n_clusters =
    n_colors = 5
    # criteria: iteration termination condition (type, max, epsilon accuracy)=
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    # attemps: number of times algorithm is initialized =
    attempts = 10
    # flags: how initial centers are taken =
    flags = cv2.KMEANS_RANDOM_CENTERS

    ## OUTPUT

    # compactness: sum of squared distance from each point to their centers (oseeef)
    # labels: array labeling from 0 to n_colors each pixel
    # centers: array of centers of clusters (each center is a color)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, attempts, flags)

    # count number of times each label appears
    _, counts = np.unique(labels, return_counts=True)

    if not return_table:
        indices = np.argsort(counts)[::-1]
        color0 = palette[indices[0]]
        color1 = palette[indices[1]]
        color2 = palette[indices[2]]
        color3 = palette[indices[3]]
        color4 = palette[indices[4]]

        return color0, color1, color2, color3, color4

    # delimiate and draw the dominant colors table
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices] / float(counts.sum())]))
    rows = np.int_(image.shape[0] * freqs)

    dom_patch = np.zeros(shape=image.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i] : rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    # resize dom_patch to a square
    dom_patch = skimage.transform.resize(dom_patch, output_shape=(226, 226, 3))

    return image, dom_patch


def draw_table(table: np.array, image: np.array = False, title: str = "table"):

    if type(image) == type(np.ones(shape=(1, 1))):
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))

        ax[0].imshow(image)
        ax[0].axis("off")

        ax[1].imshow(table, extent=[0, 100, 0, 100])
        ax[1].axis("off")
        ax[1].set_title(f"{title}")

        plt.savefig(f"{title}.png")
        plt.close()

    else:
        plt.imshow(table, extent=[0, 100, 0, 100])
        plt.axis("off")
        plt.title(f"{title}")

        plt.savefig(f"{title}.png")
        plt.close()


def pred_no_ai(image_path, method: str = "average"):

    if method == "average":

        target_vector = average_color(image_path, return_table=False)

        candidate_vectors = {
            "Art_Nouveau": np.array([159, 144, 125]),
            "Baroque": np.array([102, 85, 72]),
            "Expressionism": np.array([137, 121, 104]),
            "Impressionism": np.array([136, 122, 103]),
            "Post_Impressionism": np.array([136, 122, 103]),
            "Realism": np.array([122, 112, 95]),
            "Romanticism": np.array([123, 111, 95]),
            "Symbolism": np.array([145, 133, 119]),
        }

    if method == "dominant":

        target_vector = np.vstack(dominant_colors(image_path, return_table=False))

        candidate_vectors = {
            "Art_Nouveau": np.loadtxt('./dominant_colors/450_per_style/Art_Nouveau_Modern_dom'),
            "Baroque": np.loadtxt('./dominant_colors/450_per_style/Baroque_dom'),
            "Expressionism": np.loadtxt('./dominant_colors/450_per_style/Expressionism_dom'),
            "Impressionism": np.loadtxt('./dominant_colors/450_per_style/Impressionism_dom'),
            "Post_impressionism": np.loadtxt('./dominant_colors/450_per_style/Post_Impressionism_dom'),
            "Realism": np.loadtxt('./dominant_colors/450_per_style/Realism_dom'),
            "Romanticism": np.loadtxt('./dominant_colors/450_per_style/Romanticism_dom'),
            "Symbolism": np.loadtxt('./dominant_colors/450_per_style/Symbolism_dom'),
        }


    distance = {}
    for style in candidate_vectors.keys():
        distance[style] = np.linalg.norm(target_vector - candidate_vectors[style])

    y_pred = list(candidate_vectors)[np.argmin(list(distance.values()))]

    return y_pred


def evaluate_no_ai(test_df, method = 'average'):

    TP = 0

    for image, style in tqdm(zip(test_df['full_path'], test_df['style'])):
        y_pred = pred_no_ai(image, method = method)

        if y_pred == style:
            TP += 1

    return TP/len(test_df)



if __name__ == "__main__":
    print(
        pred_no_ai(
            "./raw_data/wikiart/in_work/Baroque/adriaen-brouwer_jan-davidszoon-de-heem.jpg", method = 'dominant'
        )
    )
    print(
        pred_no_ai(
            "./raw_data/wikiart/in_work/Impressionism/adam-baltatu_autumn-still-life.jpg", method = 'dominant'
        )
    )
    print(
        pred_no_ai(
            "./raw_data/wikiart/in_work/Impressionism/alfred-sisley_along-the-woods-in-autumn-1885.jpg", method = 'dominant'
        )
    )
    print(
        pred_no_ai(
            "./raw_data/wikiart/in_work/Symbolism/akseli-gallen-kallela_conceptio-artis-1894.jpg", method = 'dominant'
        )
    )

    print(
        pred_no_ai(
            "./raw_data/wikiart/in_work/Realism/john-singer-sargent_mrs-ernest-hills-1909.jpg", method = 'dominant'
        )
    )
