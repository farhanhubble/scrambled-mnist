import typer
from download import download_mnist_train, download_mnist_test
from augment import augment
from train import train
from test import evaluate

app = typer.Typer()

@app.command()
def download():
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
