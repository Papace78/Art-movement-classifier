import tensorflow as tf
import os

from transfer_learning.components.sorter import Sorter
from colorama import Style, Fore


class Paintings:

    def __init__(
        self,
        sorter=Sorter(),
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
    ):

        self.sorter = sorter
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split

    def sort(self):
        self.sorter.sort()

    def where(self):
        print(
            Fore.BLUE + f"\n{self.sorter.trainval_dir}" + Style.RESET_ALL
        )
        print(f"containing: {os.listdir(self.sorter.trainval_dir)}")

        print(Fore.BLUE + f"\n\n{self.sorter.test_dir}" + Style.RESET_ALL)
        print(f"containing: {os.listdir(self.sorter.test_dir)}\n\n")


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
        )

        test = tf.keras.utils.image_dataset_from_directory(
            self.sorter.test_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            interpolation="nearest",
            crop_to_aspect_ratio=True,
        )

        train_ds = train.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = validation.prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test.prefetch(tf.data.experimental.AUTOTUNE)

        return train_ds, val_ds, test_ds
