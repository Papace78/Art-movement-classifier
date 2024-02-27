import sys

from colorama import Fore, Style
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, RandomFlip, RandomZoom, RandomTranslation, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt


def initialize_model():
    model = Sequential()


    #Data Augmentation
    model.add(RandomFlip("horizontal"))
    model.add(RandomZoom(0.1))
    model.add(RandomTranslation(0.2, 0.2))
    model.add(RandomRotation(0.1))

    #CNN
    model.add(Conv2D(128, (3,3), 1, activation = 'relu', padding = 'same', input_shape = (512,512,3)))
    model.add(BatchNormalization(momentum = 0.9))
    """model.add(Conv2D(128, (3,3), 1, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization(momentum = 0.9))"""
    model.add(MaxPooling2D())
    model.add(Dropout(0.05))

    model.add(Conv2D(64, (3,3), 1, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization(momentum = 0.9))
    """model.add(Conv2D(64, (3,3), 1, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization(momentum = 0.9))"""
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))
    """model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))"""
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())

    #Dense
    model.add(Dense(128, activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization(momentum = 0.9))
    model.add(Dropout(0.4))

    model.add(Dense(8, activation = 'softmax'))

    print("✅ Model initialized")

    return model



def compile_model(model,
                   optimizer = Adam(),
                   loss = SparseCategoricalCrossentropy(),
                   metrics = ['accuracy'],
                   learning_rate = 0.001):

    model.compile(
        optimizer = Adam(learning_rate = learning_rate),
        loss = loss,
        metrics = metrics)

    print("✅ Model compiled")

    return model

def evaluate_model(model, test_generator):

    print(Fore.BLUE + f"\nEvaluating model on {len(test_generator)} rows..." + Style.RESET_ALL)

    metrics = model.evaluate(
        test_generator,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics['accuracy']

    print(f"✅ Model evaluated, accuracy: {round(accuracy*100, 2)}%")

    return metrics

def learning_curves(history):
    # plot loss
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()
