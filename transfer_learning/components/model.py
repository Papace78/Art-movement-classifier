from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomWidth,
)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


import tensorflow as tf
import os


class My_Model:
    """
    Create and compile layers:
    Data augmentation
    Preprocess
    FrozenVGG16
    Dense 1024
    Dense 512
    Dropout 0.2
    Dense n_classes softmax


    INPUT: n_classes
    OUTPUT: Model
    """

    INPUT_SHAPE = (224, 224, 3)

    def __init__(self):
        pass

    def initialize_model(self, n_classes=8):

        # -------DATA AUGMENTATION

        data_augmentation = tf.keras.Sequential(
            [
                RandomFlip("horizontal"),
                RandomRotation(0.3),
                RandomZoom(0.3),
                RandomWidth(0.2),
            ]
        )

        # --------SCALING
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

        # --------VGG16

        vgg16_model = tf.keras.applications.VGG16(
            input_shape=self.INPUT_SHAPE, include_top=False, weights="imagenet"
        )
        vgg16_model.trainable = False

        # ------CLASSIFICATION LAYERS

        global_average_layer = GlobalAveragePooling2D()
        classification_layer1 = Dense(1024, activation="relu")
        classification_layer2 = Dense(512, activation="relu")

        prediction_layer = Dense(n_classes, activation="softmax")

        # -------ARCHITECTURE

        inputs = tf.keras.Input(shape=self.INPUT_SHAPE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = vgg16_model(x, training=False)
        x = global_average_layer(x)
        x = classification_layer1(x)
        x = classification_layer2(x)
        x = Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        print(model.summary())
        print("\nnumber of trainable variable:", len(model.trainable_variables))

        return model



def train_model(model, train_dataset, validation_dataset):

    print("\nEvaluating initial loss and accuracy...")
    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("\ninitial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    ## Callbacks
    checkpoint = ModelCheckpoint(
        filepath="tl_model_v1.weights.best.hdf5", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, mode="min"
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0
    )

    print("\nnumber of trainable variable:", len(model.trainable_variables))

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=[es, checkpoint, LRreducer],
        class_weight=get_class_weights(),
    )

    model.save(f"../model/frozen.keras")

    return history


def finetune_recompile(model, finetune = 17, learning_rate= 0.00001):
    """Make top vgg16 layers trainable and recompile"""
    model.layers[4].trainable = True

    for layer in model.layers[4].layers[:finetune]:
        layer.trainable = False

    print("\nnumber of layers in the base vgg16 model:", len(model.layers[4].layers))
    print(
        "number of trainable layers in the base vgg16 model:",
        sum(
            [
                model.layers[4].layers[i].trainable == True
                for i in range(len(model.layers[4].layers))
            ]
        ),
    )
    print("\nnumber of trainable variable:", len(model.trainable_variables))



    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def finetune_model(model, history, train_dataset, validation_dataset, finetune):
    ## Callbacks
    checkpoint = ModelCheckpoint(
        filepath="tl_model_v1_finetuned.weights.best.hdf5",
        save_best_only=True,
        verbose=1,
    )
    es = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, mode="min"
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0
    )

    history_fine = model.fit(
        train_dataset,
        epochs=1000,
        initial_epochs=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=[es, checkpoint, LRreducer],
        class_weight=get_class_weights(),
    )

    model.save(f"../model/finetune.keras")

    return history_fine


def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print("\nTest accuracy:", accuracy)
    return loss, accuracy

def predict():
    #TODO
    pass


def get_class_weights():
    """individual weight = (Total_samples / n_classes * len(individual_class))"""

    trainval_dir = "raw_data/wikiart/trainval_directory"
    label_counts = {}
    for style in os.listdir(trainval_dir):
        label_counts[style] = len(os.listdir(os.path.join(trainval_dir, style)))

    total_samples = sum(label_counts.values())

    label_weights = {}
    for k, v in label_counts.items():
        label_weights[k] = total_samples / (len(label_counts) * v)

    # Get the list of class names (directory names)
    class_names = sorted(os.listdir(trainval_dir))

    # Map class names to class labels
    class_name_to_label = {i: class_name for i, class_name in enumerate(class_names)}

    class_weights = {}
    for label, style in class_name_to_label.items():
        class_weights[label] = label_weights[style]

    return class_weights
