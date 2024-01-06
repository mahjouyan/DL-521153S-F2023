# DL-521153S-F2023

## Description
Pre-training, eval & test base on mini_imagenet dataset, using ["resnet18", "resnet50", "efficientnet_b0"] and fine-tuning base on random EuroSAT dataset.

## Installation

Before running the script, make sure to install the required dependencies. Please use the following commands to create a virtual environment, activate it, and install the dependencies:

### Create a virtual environment (optional but recommended)
```bash
python -m venv venv
```

### activate the v environment
```bash
source venv/bin/activate   # on win, use `venv\Scripts\activate`
```

### install dependencies using pip
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py -h
```

### Options
- -h, --help: Show help message and exit.
- --tasks [TASKS ...]: Specify tasks to run, including train, test, and tune.
- --download: Download the dataset.
- --model {resnet10, resnet18, resnet50, efficientnet_b0}: Choose the model for training, evaluation, or testing.
- --pretrained: Load the model for training with pretrained weights.
- --tune_dataset {eurosat}: Specify the dataset for fine-tuning.
- --batch_size BATCH_SIZE: Set the batch size.
- --image_size IMAGE_SIZE: Set the image size.
- --num_epochs NUM_EPOCHS: Set the number of training epochs.
- --learning_rate LEARNING_RATE: Set the learning rate.
- --num_tune_runs NUM_TUNE_RUNS: Set the number of fine-tuning runs.

## Example
```bash
python main.py --tasks train --download --model resnet50 --pretrained --batch_size 32 --image_size 84 --num_epochs 25 --learning_rate 0.001
```

This example command will run the script with the specified options,downloading the required dataset, training and testing a ResNet50 model with pretrained weights on the MiniImageNet dataset.
