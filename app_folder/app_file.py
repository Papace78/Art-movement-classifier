from fastapi import FastAPI, File, UploadFile

import uuid
from keras.models import load_model
import tensorflow as tf

import os
import numpy as np
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt

app = FastAPI()
model = load_model(f"./model/tobedeleted.keras")

# define root
@app.get("/")
def status():
    return {"API": "connected"}


#@app.post("/file")
#async def create_file(file: Annotated[bytes, File()]):
 #   return {"file_size": len(file)}


@app.post("/predict")
async def create_upload_file(file: UploadFile= File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    with open(f"raw_data/{file.filename}", "wb") as f:
        f.write(contents)

    img_path = os.path.join('raw_data',file.filename)
    img = plt.imread(img_path)


    tensor_image = tf.convert_to_tensor(img)
    tensor_image = tf.image.resize(img,(224,224), method = 'nearest')
    tensor_image = tf.expand_dims(tensor_image, axis = 0)

    y_pred = model.predict(tensor_image)

    my_pred = str(np.argmax(y_pred))

    return my_pred


    #return FileResponse(img_path)
