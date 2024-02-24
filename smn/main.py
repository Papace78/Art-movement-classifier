
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pandas as pd
import numpy as np


from smn.data import *
from smn.model import *
from smn.preprocess import *
from smn.params import *

def main():

    data_df = get_data(image_dir = IMAGE_DIR)
    print(f"✅ Data loaded, with shape {data_df.shape}")

    data_2k = extract_from_data(data_df, images_per_style = IMAGES_PER_STYLE)
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
                          loss = SparseCategoricalCrossentropy(from_logits = True),
                          metrics = ['accuracy'],
                          learning_rate = 0.001)


    es = EarlyStopping(patience = 3, monitor = 'val_loss', restore_best_weights = True)

    checkpoint = ModelCheckpoint(
        filepath = '../model',
        save_freq = 100
    )

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(
        train_generator,
        validation_data = val_generator,
        epochs = 200,
        callbacks = [es, checkpoint]
    )

    print(f"✅ Model trained on {len(train_generator)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}")

    metrics = evaluate_model(model, test_generator=test_generator)

    learning_curves(history)

    pass
