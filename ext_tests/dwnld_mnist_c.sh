#! /bin/bash
MAIN_DATASET=${mnist_c_main_dataset:? "Error: mnist_c_main_dataset is not set"}
AUX_DATASET=${mnist_c_aux_dataset:? "Error: mnist_c_aux_dataset is not set"}

# Download the main dataset
mkdir -p data/external
wget -O data/external/mnist_c.zip $MAIN_DATASET
unzip -o data/external/mnist_c.zip -d data/external
rm data/external/mnist_c.zip

# Download the auxiliary dataset
mkdir -p data/external
wget -O data/external/mnist_c_leftovers.zip $AUX_DATASET
unzip -o data/external/mnist_c_leftovers.zip -d data/external
rm data/external/mnist_c_leftovers.zip

