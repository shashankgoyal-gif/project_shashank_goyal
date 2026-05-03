# Mushroom Classification with ResNet18

This repository contains a PyTorch-based deep learning pipeline for classifying images of mushrooms as either edible or poisonous using transfer learning on a ResNet18 model.

## Directory Structure
* `config.py`: Hyperparameters and image normalization variables.
* `dataset.py`: Custom PyTorch Dataset and DataLoader handling data augmentation.
* `model.py`: The ResNet18 model architecture modified for binary classification.
* `train.py`: The training loop and weight-saving logic.
* `predict.py`: Inference functions to classify new, unseen images.
* `interface.py`: Standardized namespace mapping for external evaluation.
* `data/`: Contains sample raw images for training and testing.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following libraries:
* PyTorch
* Torchvision
* Pillow

You can install the required packages using:
```bash
pip install torch torchvision Pillow
