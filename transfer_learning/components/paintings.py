import tensorflow as tf
import os

from transfer_learning.components.sorter import Sorter
from colorama import Style, Fore


class Paintings:
    """
    Create three tf dataset (train/val/test) from directory
    Using method .fetch()

    INPUT:
    Sorter - in order to get trainval and test directories path to import from.
    batch_size
    image_size = (224,224) by default for vgg16
    validation split between train and val

    OUTPUT:
    batched and prefetched train_ds val_ds test_ds
    """

    def __init__(
        self,
        sorter=Sorter(),
        batch_size=32,
        image_size=(224,224),
        validation_split=0.2,
    ):

        self.sorter = sorter
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split

    def sort(self):
        self.sorter.sort()

    def fetch(self):
        print("\n\nFetching:")
        self.where()

        train, validation = tf.keras.utils.image_dataset_from_directory(
            self.sorter.trainval_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            validation_split=self.validation_split,
            subset="both",
            seed=42,
            interpolation="nearest",
            crop_to_aspect_ratio=True,
            shuffle=True
        )

        test = tf.keras.utils.image_dataset_from_directory(
            self.sorter.test_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            interpolation="nearest",
            crop_to_aspect_ratio=True,
            shuffle=True
        )

        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train.prefetch(buffer_size=AUTOTUNE)
        val_ds = validation.prefetch(buffer_size=AUTOTUNE)
        test_ds = test.prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, test_ds

    def where(self):
        print(Fore.BLUE + f"\nFrom folder: {self.sorter.trainval_dir}," + Style.RESET_ALL)
        print(f"containing: {os.listdir(self.sorter.trainval_dir)}")

        print(Fore.BLUE + f"\n\nFrom folder: {self.sorter.test_dir}," + Style.RESET_ALL)
        print(f"containing: {os.listdir(self.sorter.test_dir)}\n\n")
