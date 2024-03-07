from transfer_learning.components.utils import (
    sort_files,
    get_data,
    create_model,
    learning_curves,
)
from transfer_learning.components.model import (
    train_model,
    finetune_model,
    evaluate_model,
)

from keras.models import load_model

import os
from colorama import Fore, Style
import numpy as np


def classification(
    finetune=17,
    batch_size=32,
    image_size=(224, 224),
    validation_split=0.2,
    n_classes=8,
):
    print("\nFinetune =", finetune)

    # CREATE TRAINVAL and TEST DIRECTORIES
    if not "trainval_directory" in os.listdir("raw_data/wikiart/"):
        sort_files()

    # FETCH THE DATA AND LOAD THEM IN A DATASET
    train, val, test = get_data(
        batch_size=batch_size, image_size=image_size, validation_split=validation_split
    )

    # CREATE MODEL:
    #   data augmentation
    #   preprocess_input
    #   frozen vgg16
    #   two dense layer
    #   dropout 0.2
    #   classification layer


    vgg16 = create_model(n_classes=n_classes)


    # TRAIN MODEL on frozen vgg16
    print(Fore.BLUE + f"\nTraining model on {len(train)} rows..." + Style.RESET_ALL)
    history = train_model(vgg16, train, val)

    print(
        f"✅ Model trained on {len(train)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}"
    )

    # FINETUNE MODEL
    if finetune:
        print(
            Fore.BLUE + f"\nFinetuning model on {len(train)} rows..." + Style.RESET_ALL
        )
        history_fine = finetune_model(
            vgg16,
            history=history,
            train_dataset=train,
            validation_dataset=val,
            finetune=finetune,
        )

    # EVALUATE MODEL
    print(Fore.BLUE + f"\nTesting model on {len(test)} rows..." + Style.RESET_ALL)
    accuracy = evaluate_model(vgg16, test)
    print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")

    # PLOT LEARNING CURVES
    if finetune:
        learning_curves(history_fine, title="finetunening")
    else:
        learning_curves(history, title="feature_extraction")

    print(accuracy)


def prediction(test_dataset):

    model = load_model(f'../model/finetune.keras')

    prediction = model.predict(test_dataset)

    return prediction
