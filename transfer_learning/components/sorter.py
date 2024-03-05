"""
train/test split on local drive
We will then split train/val directly inside tf.image_dataset_from_directory

End result:
Three folders "train" "validation" "test"
Each folder will have all 8 subfolders, aka labels,
containting the paintings.jpg.
"""

import os
import shutil
import random


class Sorter:
    def __init__(
        self,
        source_dir="raw_data/wikiart/in_work",
        trainval_dir="raw_data/wikiart/trainval_directory",
        test_dir="raw_data/wikiart/test_directory",
        split_ratio=0.9,
    ) -> None:

        self.source_dir = source_dir
        self.trainval_dir = trainval_dir
        self.test_dir = test_dir

        self.split_ratio = split_ratio

    # Function to create directory structure
    def create_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Function to move files to appropriate directories
    def move_files(self, source, destination, files):
        for file in files:
            shutil.move(os.path.join(source, file), os.path.join(destination, file))

    # Iterate through subfolders
    def sort(self):

        # create directories
        self.create_dir(self.trainval_dir)
        self.create_dir(self.test_dir)

        for label in os.listdir(self.source_dir):
            self.create_dir(os.path.join(self.trainval_dir, label))
            self.create_dir(os.path.join(self.test_dir, label))

            label_dir = os.path.join(self.source_dir, label)

            # Skip if it's not a directory
            if not os.path.isdir(label_dir):
                continue

            # Get list of files and shuffle it
            files = os.listdir(label_dir)
            random.shuffle(files)

            # Calculate split indices
            trainval_split = int(len(files) * self.split_ratio)

            # Split the files
            trainval_files = files[:trainval_split]
            test_files = files[trainval_split:]

            # Move files to respective directories
            self.move_files(
                source=label_dir,
                destination=os.path.join(self.trainval_dir, label),
                files=trainval_files,
            )

            self.move_files(
                source=label_dir,
                destination=os.path.join(self.test_dir, label),
                files=test_files,
            )

            print(f"✅ sorted {label}")
