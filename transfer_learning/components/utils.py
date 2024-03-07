from transfer_learning.components.paintings import Paintings
from transfer_learning.components.model import My_Model
from transfer_learning.components.sorter import Sorter

import matplotlib.pyplot as plt
import os



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



def create_model(n_classes=8):

    my_model = My_Model()

    my_model = my_model.initialize_model(
        n_classes=n_classes
    )

    return my_model


def learning_curves(history, title = 'myman.png'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(acc,label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epcoh')

    plt.savefig(title)
    plt.close()
