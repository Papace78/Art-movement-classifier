import os
import typer

from keras.models import load_model
from colorama import Fore, Style

from core import classification
from vgg16 import finetune_recompile, finetune_model, evaluate_model, predict_model
from params import FINETUNE, BATCH_SIZE, N_CLASSES, LR, IMAGE_SIZE
from plots import learning_curves
from tfpaintings import Paintings


app = typer.Typer(add_completion=False)


@app.command()
def classify(
    finetune: int = typer.Option(FINETUNE, help="Number of vgg16 layers to train."),
    batch_size: int = typer.Option(BATCH_SIZE, help="Paintings per batch."),
    validation_split: float = typer.Option(0.2, help="Split raito."),
    n_classes: int = typer.Option(N_CLASSES, help="Number of movements to classify."),
    learning_rate: float = typer.Option(LR, help="LR for frozen, LR/10 for finetune"),
):
    """
    Train a new classifier
    """

    classification(
        finetune=finetune,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        validation_split=validation_split,
        n_classes=n_classes,
        learning_rate=learning_rate,
    )


@app.command()
def predict(
    image_path: str = typer.Option(
        "raw_data/wikiart/test_directory", help="Path to the image"
    )
):
    """
    Get prediction from latest finetuned model
    """
    model = load_model(os.path.join("model", "finetune_17"))

    y_array, y_pred, y_name = predict_model(model, image_path=image_path)
    print("\ny_array:", y_array)
    print("\ny_pred, y_name:", [y_pred, y_name])

    return {"prediction": [y_pred, y_name]}


@app.command()
def finetuning(
    finetune: int = typer.Option(
        FINETUNE, help="Finetune the model, skipping the basic frozen training"
    )
):
    """
    Load and finetune a model pretrained on your own dataset
    """

    #LOAD MODEL
    print("Finetune =", finetune)
    model_frozen = load_model(os.path.join("model", "frozen"))
    print("✅ Trained model loaded")
    model_fine = finetune_recompile(model_frozen, finetune=finetune)

    #FETCH DATA
    my_paintings = Paintings()
    train, val, test = my_paintings.fetch()

    #TRAIN MODEL
    print(Fore.BLUE + f"\nFinetuning model on {len(train)} rows..." + Style.RESET_ALL)
    history_fine = finetune_model(
        model_fine, train_dataset=train, validation_dataset=val, finetune=finetune
    )

    # PLOT LEARNING CURVES
    learning_curves(history_fine, title=f"finetuned_{finetune}")
    print(f"✅ Saved learning_curves")

    # EVALUATE MODEL
    print(Fore.BLUE + f"\nTesting fine model on {len(test)} rows..." + Style.RESET_ALL)
    loss, accuracy = evaluate_model(model_fine, test)
    print(f"✅ Model tested, accuracy: {round(accuracy*100, 2)}%")

    return history_fine


app()
