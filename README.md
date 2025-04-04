# Scramble MNIST 
Can we train a model to learn to classify noise along with regular patterns?

This repository trains a barebones neural network on a dataset made up of regular MNIST images and their scrambled copies. The idea is to see if the network can learn to classify the noise into its own class. For this purpose we first create an augmented train dataset and then train a convnet on this dataset. 

## Setup
- Install pyenv for isolated Python installation
- Install `Poetry` for dependency management
- Clone this repo
- Activate your Python interpreter, for example with `pyenv local 3.12`
- Create a `Poetry` environment with `poetry install`
- Activate the environment with `eval $(poetry env activate)`
- Create an account on Kaggle and get an API key
- Place the API key in a new directory `.secrets/kaggle.json`
- Run `dvc repro` to run the full pipeline. This will:
    - Download the MNIST dataset as CSV files
    - Convert each row from the CSV files to an image file and organize the image files into one subdirectory per label, which is easier to load with Pytorch
    - Augment the training dataset by scrambling a fraction of the dataset and assigning it a distinct label
    - Train a convnet to learn all 11 labels
    - Test the convnet on the standard 10 labels 
    - Download the [MNIST-C][MNIST_C] dataset that contains MNIST-derived images corrupted by various types of noise
    - Evaluate the model's performance on MNIST_C test set
- Optionally check out `exp_nonoise` branch and
    - Run `dvc repro` to retrain the convnet on the orignial MNIST set by reducing the fraction of augmented images to just 0.005. This reruns all the evaluations from above
    - Compare the report.txt file from the `main` and `exp_nonoise`


> If you face any problems or have questions, create a new issue here on Github.

[MNIST_C]: https://github.com/google-research/mnist-c
