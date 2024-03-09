from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomWidth,
)

import tensorflow as tf


class My_Model:
    def __init__(self):
        pass

    def initialize_model(
        self, input_shape=(224, 224, 3), n_classes=8, optimizer=Adam(), fine_tune=0
    ):
        """
        Compiles a model integrated with VGG16 pretrained layers
        input_shape: tuple - the shape of input images (width, height, channels)
        n_classes: int - number of classes for the output layer
        optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        fine_tune: int - The number of pre-trained layers to unfreeze.
                    If set to 0, all pretrained layers will freeze during training
        """

        # -------DATA AUGMENTATION

        data_augmentation = tf.keras.Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.2),
                RandomZoom(0.2),
                RandomWidth(0.2),
            ]
        )

        # ------VGG16

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = VGG16(
            include_top=False, weights="imagenet", input_shape=input_shape
        )

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # ------CLASSIFICATION LAYERS

        # flatten = Flatten(name='flatten')
        global_average_layer = GlobalAveragePooling2D()
        classification_layer1 = Dense(256, activation="relu")
        classification_layer2 = Dense(128, activation="relu")
        dropout = Dropout(0.2)

        prediction_layer = Dense(n_classes, activation="softmax")

        # -------ARCHITECTURE
        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        inputs = Input(shape=input_shape)
        bottom_model = data_augmentation(inputs)
        bottom_model = preprocess_input(bottom_model)
        mid_model = conv_base(bottom_model)
        top_model = global_average_layer(mid_model)
        top_model = classification_layer1(top_model)
        top_model = classification_layer2(top_model)
        top_model = dropout(top_model)
        output_layer = prediction_layer(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=inputs, outputs=output_layer)

        # -------COMPILE
        # Compiles the model for training.
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
