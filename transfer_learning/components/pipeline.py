from transfer_learning.components.paintings import Paintings
from transfer_learning.components.model import My_Model
from transfer_learning.components.sorter import Sorter
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

from colorama import Fore, Style

from keras.optimizers import Adam


def sort_files(source_dir="raw_data/wikiart/in_work"):
    sort = Sorter(source_dir)
    sort.sort()

    return print("✅ All sorted")


def get_data(batch_size=32, image_size=(224, 224), validation_split=0.2):

    my_paintings = Paintings(
        sorter=Sorter(),
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
    )

    return my_paintings.fetch()


def create_model(input_shape=(224, 224, 3), n_classes=8, optimizer=Adam(), fine_tune=0):

    my_model = My_Model()

    my_model = my_model.initialize_model(
        input_shape=input_shape,
        n_classes=n_classes,
        optimizer=optimizer,
        fine_tune=fine_tune,
    )

    return my_model


def train_model(model, train, val):
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

    print(Fore.BLUE + f"\ntraining model on {len(train)} rows..." + Style.RESET_ALL)

    history = model.fit(
        train,
        validation_data=val,
        epochs=1000,
        callbacks=[es, checkpoint, LRreducer],
    )

    return history


def evaluate_model(model, test):

    print(Fore.BLUE + f"\nEvaluating model on {len(test)} rows..." + Style.RESET_ALL)

    metrics = model.evaluate(
        test,
        verbose=0,
        # callbacks=None,
        return_dict=True,
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy*100, 2)}%")

    return metrics


def learning_curves(history):
    # plot loss
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="test")
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="orange", label="test")
    plt.legend()
    # save plot to file
    plt.savefig("entonnoir_doublecouche.png")
    plt.close()
