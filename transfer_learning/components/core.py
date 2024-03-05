from pipeline import (
    sort_files,
    get_data,
    create_model,
    train_model,
    evaluate_model,
    learning_curves,
)
from keras.optimizers import Adam
import os
from colorama import Fore, Style
import numpy as np


def classification():

    if not "trainval_directory" in os.listdir("raw_data/wikiart/"):
        sort_files()

    train, val, test = get_data(
        batch_size=32, image_size=(224, 224), validation_split=0.2
    )

    vgg16 = create_model(
        input_shape=(224, 224, 3), n_classes=8, optimizer=Adam(), fine_tune=0
    )

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    history = train_model(vgg16, train, val)

    print(
        f"✅ Model trained on {len(train)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}"
    )

    metrics = evaluate_model(vgg16, test)
    learning_curves(history)

    print(metrics)

if __name__ == '__main__':
    classification()
