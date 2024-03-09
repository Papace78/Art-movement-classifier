from transfer_learning.components.model import (
    train_model,
    finetune_recompile,
    finetune_model,
    evaluate_model,
)
from transfer_learning.components.plots import learning_curves


from transfer_learning.components.sorter import Sorter
from transfer_learning.components.paintings import Paintings
from transfer_learning.components.model import My_Model

import os
import tensorflow as tf
from colorama import Fore, Style
import numpy as np

SOURCE_DIR = "raw_data/wikiart/"


def classification(
    finetune=17,
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    n_classes=8,
    learning_rate=0.0001,
):
    print("\nFinetune =", finetune)

    # CREATE TRAINVAL and TEST DIRECTORIES
    if not "trainval_directory" in os.listdir(SOURCE_DIR):
        sort = Sorter(SOURCE_DIR)
        sort.sort()
        print("✅ All sorted")

    # FETCH THE DATA AND LOAD THEM IN A DATASET
    my_paintings = Paintings(
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
    )

    train, val, test = my_paintings.fetch()

    # CREATE MODEL:
    #   data augmentation
    #   preprocess_input
    #   frozen vgg16
    #   two dense layer
    #   dropout 0.2
    #   classification layer

    vgg16 = My_Model()
    vgg16 = vgg16.initialize_model(n_classes=n_classes)
    vgg16.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # TRAIN MODEL on frozen vgg16
    print(Fore.BLUE + f"\nTraining model on {len(train)} rows..." + Style.RESET_ALL)
    history = train_model(vgg16, train, val)

    print(
        f"✅ Model trained on {len(train)} rows with min val accuracy:\
        {round(np.min(history.history['val_accuracy']), 2)}"
    )



    # FINETUNE MODEL
    if finetune:
        print(
            Fore.BLUE + f"\nFinetuning model on {len(train)} rows..." + Style.RESET_ALL
        )
        vgg16_fine = finetune_recompile(
            vgg16, finetune=finetune, learning_rate=learning_rate / 10
        )
        history_fine = finetune_model(
            vgg16_fine,
            history=history,
            train_dataset=train,
            validation_dataset=val,
            finetune=finetune,
        )

        # EVALUATE MODEL
        print(Fore.BLUE + f"\nTesting fine model on {len(test)} rows..." + Style.RESET_ALL)
        loss, accuracy = evaluate_model(vgg16_fine, test)
        print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")

        # PLOT LEARNING CURVES
        learning_curves(history_fine, title="finetunening")

    else:
        print(Fore.BLUE + f"\nTesting model on {len(test)} rows..." + Style.RESET_ALL)
        loss, accuracy = evaluate_model(vgg16, test)
        print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")


        learning_curves(history, title="feature_extraction")

    return loss, accuracy
