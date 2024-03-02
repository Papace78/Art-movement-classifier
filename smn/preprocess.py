import tensorflow as tf

"""
Reshape images + and rescale between 0 and 1
"""


def process_image(image_path, label, image_size = (512,512)):
    image = tf.io.read_file(image_path[0])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size, method = 'nearest', preserve_aspect_ratio=False)
    image = image / 255

    return image, label
