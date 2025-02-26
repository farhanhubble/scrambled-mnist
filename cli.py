import typer
from download import download_mnist_train, download_mnist_test
from augment import augment
from train import train
from test import evaluate

app = typer.Typer()

@app.command()
def download_train():
    download_mnist_train()

@app.command()
def download_test():
    download_mnist_test()

@app.command()
def augment_data():
    augment()

@app.command()
def train_model():
    train()

@app.command()
def evaluate_model():
    evaluate()

if __name__ == "__main__":
    app()