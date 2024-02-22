from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def initialize_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform', input_shape = (512,512,3)))
    model.add(Conv2D(32, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3,3), 1, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D())

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
