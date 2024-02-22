import os

IMAGE_DIR = os.environ.get('IMAGE_DIR')

IMAGES_PER_STYLE = int(os.environ.get('IMAGES_PER_STYLE'))


DATASET_PATH = os.environ.get('DATASET_PATH')

GCP_REGION= os.environ.get("GCP_REGION")
GCP_PROJECT = os.environ.get("GCP_PROJECT")

BUCKET_NAME = os.environ.get("BUCKET_NAME")

INSTANCE = os.environ.get("INSTANCE")
