
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from colorama import Fore, Style
import tensorflow as tf
import pandas as pd
import numpy as np


from smn.data import get_data, extract_from_data, split_data, generate_generators
from smn.model import initialize_model, compile_model, evaluate_model, learning_curves
from smn.params import IMAGE_DIR, IMAGES_PER_STYLE

def main():

    data_df = get_data(image_dir = IMAGE_DIR)
    print(f"✅ Data loaded, with shape {data_df.shape}")

    data_2k = extract_from_data(data_df, images_per_style = 4000)
    print(f"✅ Data extracted, with shape {data_2k.shape}")


    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_2k,
                                                                train_size = 0.9,
                                                                val_size = 0.1,
                                                                random_state = 42)

    train_generator, val_generator, test_generator = generate_generators(
        X_train = X_train,
        X_val = X_val,
        X_test = X_test,
        y_train = y_train,
        y_val = y_val,
        y_test = y_test)


    model = initialize_model()
    model = compile_model(model,
                          optimizer = Adam(),
                          loss = SparseCategoricalCrossentropy(),
                          metrics = ['accuracy'],
                          learning_rate = 0.001)



    es = EarlyStopping(patience = 15, monitor = 'val_loss', restore_best_weights = True)

    LRreducer = ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience=3, verbose=1, min_lr=0)

    checkpoint = ModelCheckpoint(
        filepath = '../model',
        save_freq = 200
    )

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(
        train_generator,
        validation_data = val_generator,
        epochs = 1000,
        callbacks = [es, checkpoint, LRreducer]
    )

    print(f"✅ Model trained on {len(train_generator)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}")

    metrics = evaluate_model(model, test_generator=test_generator)

    learning_curves(history)

    pass
