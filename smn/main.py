from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tqdm import tqdm
from colorama import Fore, Style

import tensorflow as tf
import pandas as pd
import numpy as np


from smn.data import get_data, extract_from_data, split_data, generate_generators
from smn.model import initialize_model, compile_model, evaluate_model, learning_curves
from smn.no_ai import average_color, dominant_colors, draw_table, evaluate_no_ai
from smn.params import IMAGE_DIR


WITHOUT_AI = False
IMAGES_PER_STYLE = 4000
N_LABELS = 8


def main():

    data_df = get_data(image_dir=IMAGE_DIR)
    print(f"✅ Data loaded, with shape {data_df.shape}")

    data_extract = extract_from_data(data_df, images_per_style=IMAGES_PER_STYLE)
    print(f"✅ Data extracted, with shape {data_extract.shape}")

    if WITHOUT_AI:

        #GENERATE OUR AVERAGE AND DOMINANT TABLES
        '''
        change_of_style_index = np.arange(
            0, len(data_extract), IMAGES_PER_STYLE
        ).astype(int)
        canva = np.ones(shape=(1, 1, 3))

        # average_color_per_style = {}
        # average_col = []

        dominant_colors_per_style = {}
        dom_col0 = []
        dom_col1 = []
        dom_col2 = []
        dom_col3 = []
        dom_col4 = []

        for i in tqdm(range(len(data_extract))):

            if i in change_of_style_index:
                # we change style
                if i == 0:
                    style = data_extract["style"].iloc[i]

                else:

                    print(style)

                    # average_table = (canva*np.mean(average_col, axis = 0)).astype(int)
                    # average_color_per_style[style] = average_table
                    # average_col = []

                    dom_col0 = np.mean(dom_col0, axis=0).astype(int)
                    dom_col1 = np.mean(dom_col1, axis=0).astype(int)
                    dom_col2 = np.mean(dom_col2, axis=0).astype(int)
                    dom_col3 = np.mean(dom_col3, axis=0).astype(int)
                    dom_col4 = np.mean(dom_col4, axis=0).astype(int)

                    np.savetxt(
                        f"{style} dom",
                        np.vstack(
                            (dom_col0, dom_col1, dom_col2, dom_col3, dom_col4)
                        ).astype(int),
                    )

                    dominant_palette = np.vstack(
                        (
                            canva * dom_col0,
                            canva * dom_col1,
                            canva * dom_col2,
                            canva * dom_col3,
                            canva * dom_col4,
                        )
                    )

                    dominant_colors_per_style[style] = dominant_palette.astype(int)

                    dom_col0 = []
                    dom_col1 = []
                    dom_col2 = []
                    dom_col3 = []
                    dom_col4 = []

                    style = data_extract["style"].iloc[i]

            image_path = data_extract["full_path"].iloc[i]

            # average_col.append(average_color(image_path, return_table = False))

            color0, color1, color2, color3, color4 = dominant_colors(
                image_path, return_table=False
            )
            dom_col0.append(color0)
            dom_col1.append(color1)
            dom_col2.append(color2)
            dom_col3.append(color3)
            dom_col4.append(color4)

        # average_table = (np.ones(shape = (512,512,3))*np.mean(average_col, axis = 0)).astype(int)
        # average_color_per_style['Symbolism'] = average_table

        dom_col0 = np.mean(dom_col0, axis=0).astype(int)
        dom_col1 = np.mean(dom_col1, axis=0).astype(int)
        dom_col2 = np.mean(dom_col2, axis=0).astype(int)
        dom_col3 = np.mean(dom_col3, axis=0).astype(int)
        dom_col4 = np.mean(dom_col4, axis=0).astype(int)

        np.savetxt(
            f"{style} dom",
            np.vstack((dom_col0, dom_col1, dom_col2, dom_col3, dom_col4)).astype(int),
        )

        dominant_palette = np.vstack(
            (
                canva * dom_col0,
                canva * dom_col1,
                canva * dom_col2,
                canva * dom_col3,
                canva * dom_col4,
            )
        )

        dominant_colors_per_style["Symbolism"] = dominant_palette.astype(int)

        for k, v in dominant_colors_per_style.items():
            draw_table(table=v, title=f"{k} dom")  # {np.unique(v,axis = 0)}

        return print("images saved")
        '''


        #EVALUATE EACH MODEL

        data_test = extract_from_data(data_df, images_per_style=100)
        print(f"✅ Data test, with shape {data_test.shape}")

        accuracy_average = round(evaluate_no_ai(data_test, method = 'average') * 100, 2)
        accuracy_dominant = round(evaluate_no_ai(data_test, method = 'dominant') * 100, 2)


        return print({'accuracy_average %' : accuracy_average,
                'accuracy_dominant %' : accuracy_dominant})




    # --------------GENERATE GENERATORS--------------

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data_extract, train_size=0.9, val_size=0.1, random_state=42
    )

    train_generator, val_generator, test_generator = generate_generators(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # --------------CREATE MODEL--------------

    model = initialize_model()
    model = compile_model(
        model,
        optimizer=Adam(),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        learning_rate=0.001,
    )

    es = EarlyStopping(patience=15, monitor="val_loss", restore_best_weights=True)

    LRreducer = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=0
    )

    checkpoint = ModelCheckpoint(filepath="../model", save_freq=200)

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    # --------------TRAIN--------------

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1000,
        callbacks=[es, checkpoint, LRreducer],
    )

    print(
        f"✅ Model trained on {len(train_generator)} rows with min val accuracy: {round(np.min(history.history['val_accuracy']), 2)}"
    )

    # --------------EVALUATE--------------

    metrics = evaluate_model(model, test_generator=test_generator)

    learning_curves(history)

    pass
