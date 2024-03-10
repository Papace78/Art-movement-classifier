import os
import tensorflow as tf
import numpy as np

from colorama import Fore, Style
from transfer_learning.components.model import (
    train_model,
    finetune_recompile,
    finetune_model,
    evaluate_model,
)
from transfer_learning.components.params import (
    SOURCE_DIR,
    BATCH_SIZE,
    FINETUNE,
    IMAGE_SIZE,
    N_CLASSES,
    LR,
)
from transfer_learning.components.sorter import Sorter
from transfer_learning.components.paintings import Paintings
from transfer_learning.components.model import My_Model
from transfer_learning.components.plots import learning_curves


def classification(
    finetune=FINETUNE,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    validation_split=0.2,
    n_classes=N_CLASSES,
    learning_rate=LR,
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

        # PLOT LEARNING CURVES
        learning_curves(history_fine, title="finetunening")
        print(f"✅ Saved learning_curves")

        # EVALUATE MODEL
        print(
            Fore.BLUE + f"\nTesting fine model on {len(test)} rows..." + Style.RESET_ALL
        )
        loss, accuracy = evaluate_model(vgg16_fine, test)
        print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")

    else:
        learning_curves(history, title="feature_extraction")
        print(f"✅ Saved learning_curves")

        print(Fore.BLUE + f"\nTesting model on {len(test)} rows..." + Style.RESET_ALL)
        loss, accuracy = evaluate_model(vgg16, test)
        print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")

    return loss, accuracy
