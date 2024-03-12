import tensorflow as tf
import uuid
import os

from fastapi import FastAPI, File, UploadFile

from transfer_learning.components.vgg16 import predict_model
from transfer_learning.components.params import FINETUNE

app = FastAPI()
model = tf.keras.models.load_model(os.path.join('model',f'frozen_charlax_valacc_61'))


# define root
@app.get("/")
def status():
    return {"API": "connected"}


# @app.post("/file")
# async def create_file(file: Annotated[bytes, File()]):
#   return {"file_size": len(file)}


@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(os.path.join("uploaded_painting",file.filename), "wb") as f:
        f.write(contents)

    img_path = os.path.join("uploaded_painting", file.filename)

    _ , y_label ,y_name = predict_model(model, img_path)
    y_label = int(y_label)

    return {'mypred' : [y_label, y_name]}
