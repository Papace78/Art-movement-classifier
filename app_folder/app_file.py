from fastapi import FastAPI
##import pickle
from transfer_learning.components.test_api import forecast
from transfer_learning.components.core import classification, prediction

app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

@app.get("/predict")
def predict(X=5):

    #model = pickle.load_model()
    prediction = forecast(X)

    return {'forecast': prediction}

@app.get("/train")
def training(finetune=17,
        batch_size=32,
        image_size=(224,224),
        validation_split=0.2,
        n_classes=8):

    return classification(
        finetune=finetune,
        batch_size=batch_size,
        image_size=(224,224),
        validation_split=validation_split,
        n_classes=n_classes,
    )


@app.get("/predict_real")
def predict_real(test_ds):

    predict = prediction(test_dataset=test_ds)

    return {'prediction' : predict}
