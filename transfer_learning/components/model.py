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

    INPUT_SHAPE = (224, 224, 3)
    LEARNING_RATE = 0.0001

    def __init__(self):
        pass

    def initialize_model(self, n_classes=8):
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
                RandomRotation(0.1),
                RandomZoom(0.3),
                RandomWidth(0.2),
            ]
        )

        # --------SCALING
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

        # --------VGG16

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        vgg16_model = tf.keras.applications.VGG16(
            input_shape=self.INPUT_SHAPE, include_top=False, weights="imagenet"
        )
        vgg16_model.trainable = False


        # ------CLASSIFICATION LAYERS

        # flatten = Flatten(name='flatten')
        global_average_layer = GlobalAveragePooling2D()
        classification_layer1 = Dense(256, activation="relu")
        classification_layer2 = Dense(128, activation="relu")

        prediction_layer = Dense(n_classes, activation="softmax")

        # -------ARCHITECTURE
        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        inputs = tf.keras.Input(shape=self.INPUT_SHAPE)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = vgg16_model(x, training=False)
        x = global_average_layer(x)
        x = Dropout(0.2)(x)
        x = classification_layer1(x)
        x = classification_layer2(x)
        x = Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        print(model.summary())
        # only one layer trainable 4104 parameters divided into two cat: weight and bias
        print("\nnumber of trainable variable:", len(model.trainable_variables))

        # -------COMPILE
        # Compiles the model for training.

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        )

        return model


def train_model(model, train_dataset, validation_dataset):

    print('\nEvaluating initial loss and accuracy...')
    initial_epochs = 1000
    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("\ninitial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    ## Callbacks
    checkpoint = ModelCheckpoint(
        filepath="tl_model_v1.weights.best.hdf5", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, mode="min"
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0
    )

    print("\nnumber of trainable variable:", len(model.trainable_variables))

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=initial_epochs,
        callbacks=[es, checkpoint, LRreducer],
        class_weight = get_class_weights()
    )

    model.save(f'../model/frozen.keras')

    return history


def finetune_model(model, history, train_dataset, validation_dataset, finetune):

    '''Make top vgg16 layers trainable and recompile'''
    model.layers[4].trainable = True


    for layer in model.layers[4].layers[:finetune]:
        layer.trainable = False

    print("\nnumber of layers in the base vgg16 model:", len(model.layers[4].layers))
    print("number of trainable layers in the base vgg16 model:", len(model.layers[4].layers)-finetune)
    print("\nnumber of trainable variable:", len(model.trainable_variables))

    fine_tune_epochs = 1000

    ## Callbacks
    checkpoint = ModelCheckpoint(
        filepath="tl_model_v1_finetuned.weights.best.hdf5", save_best_only=True, verbose=1
    )
    es = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, mode="min"
    )
    LRreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0
    )

    model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        )

    history_fine = model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        initial_epochs=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=[es, checkpoint, LRreducer],
        class_weight = get_class_weights()
    )

    model.save(f'../model/finetune.keras')

    return history_fine


def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print("\nTest accuracy:", accuracy)
    return accuracy



def get_class_weights():
    trainval_dir = "raw_data/wikiart/trainval_directory"
    label_counts={}
    for style in os.listdir(trainval_dir):
        label_counts[style] = len(os.listdir(os.path.join(trainval_dir,style)))

    total_samples = sum(label_counts.values())

    label_weights = {}
    for k, v in label_counts.items():
        label_weights[k] = total_samples / (len(label_counts) * v)

    # Get the list of class names (directory names)
    class_names = sorted(os.listdir(trainval_dir))

    # Map class names to class labels
    class_name_to_label = {i: class_name for i, class_name in enumerate(class_names)}

    class_weights={}
    for label, style in class_name_to_label.items():
        class_weights[label] = label_weights[style]

    return class_weights
