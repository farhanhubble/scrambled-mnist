import typer
from config import config
from convert import csv_to_images
from download import download_mnist_train, download_mnist_test, set_auth
from augment import augment
from train import train
from test import evaluate

app = typer.Typer()


@app.command()
def convert():
    csv_to_images(
        config.data_dir + "/train/raw/+" + config.kaggle_mnist_train_file,
        config.data_dir + "train/converted",
    )
    csv_to_images(
        config.data_dir + "/test/raw/+" + config.kaggle_mnist_test_file,
        config.data_dir + "test/converted",
    )


@app.command()
def download():
    set_auth()
    download_mnist_train()
    download_mnist_test()


@app.command()
def data_augment():
    augment()


@app.command()
def run_train():
    train()


@app.command()
def run_test():
    evaluate()


if __name__ == "__main__":
    app()
