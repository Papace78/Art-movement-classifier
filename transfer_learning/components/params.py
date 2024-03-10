import os


SOURCE_DIR = os.path.join("ds")

TRAINVAL_DIR = os.path.join("paintings", "trainval_dir")
TEST_DIR = os.path.join("paintings", "test_dir")
SPLIT_RATIO = 0.9

CLASS_NAMES = [
    "abstract",
    "color_field_painting",
    "cubism",
    "expressionism",
    "impressionism",
    "realism",
    "renaissance",
    "romanticism",
]

N_CLASSES = len(CLASS_NAMES)


BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = (224, 224, 3)

FINETUNE = 17
LR = 0.0001
