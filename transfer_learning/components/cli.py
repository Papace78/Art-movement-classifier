import typer

from transfer_learning.components.core import classification, prediction


app = typer.Typer(add_completion=False)

@app.command()
def classify(
    finetune: int = typer.Option(17, help="Number of vgg16 layers to train."),
    batch_size: int = typer.Option(32, help="Paintings per batch."),
    validation_split: float = typer.Option(0.2, help="Split raito."),
    n_classes: int = typer.Option(8, help="Number of movements to classify."),
):

    classification(
        finetune=finetune,
        batch_size=batch_size,
        image_size=(224,224),
        validation_split=validation_split,
        n_classes=n_classes,
    )

@app.command()
def predict(test_ds = 'raw_data/wikiart/test_directory'):

    predict = prediction(test_dataset=test_ds)

    return {'prediction' : predict}

app()
