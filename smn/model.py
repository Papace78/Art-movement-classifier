from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from colorama import Fore, Style
import time

def initialize_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform', padding = 'same', input_shape = (512,512,3)))
    #model.add(Conv2D(32, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform', padding = 'same'))
    model.add(MaxPooling2D())

    """model.add(Conv2D(64, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())"""

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(13, activation = 'softmax'))

    return model

def compile_model(model,
                   optimizer = Adam(),
                   loss = SparseCategoricalCrossentropy(from_logits=True),
                   metrics = ['accuracy'],
                   learning_rate = 0.001):

    model.compile(
        optimizer = Adam(learning_rate = learning_rate),
        loss = loss,
        metrics = ['accuracy'])

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

    print(f"✅ Model evaluated, MAE: {round(accuracy, 2)}")

    return metrics
