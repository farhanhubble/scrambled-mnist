import typer
from config import config
from convert import csv_to_images
from download import download_mnist_train, download_mnist_test, set_auth
from augment import augment as augment_data
from seed import seed
from train import train as do_train
from test import evaluate as do_evaluate

seed()
app = typer.Typer()


@app.command(help="Convert MNIST CSV data to PNG images.")
def convert():
    csv_to_images(
        config.data_dir + "/train/raw/" + config.kaggle_mnist_train_file,
        config.data_dir + "/train/converted",
    )
    csv_to_images(
        config.data_dir + "/test/raw/" + config.kaggle_mnist_test_file,
        config.data_dir + "/test/converted",
    )


@app.command(help="Download MNIST data from Kaggle.")
def download():
    set_auth()
    download_mnist_train()
    download_mnist_test()


@app.command(help="Augment MNIST data with scrambled images")
def augment():
    augment_data(
        config.data_dir + "/train/converted",
        config.data_dir + "/train/augmented",
        config.scrambled_copies,
        config.scramble_fraction,
    )


@app.command(help="Train the model")
def train():
    do_train()


@app.command(help="Evaluate the model")
def test():
    do_evaluate()


if __name__ == "__main__":
    app()
