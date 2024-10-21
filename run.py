#@title Main run
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import utilities
import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Process arguments e, b, and i.")

# Prompt user for additional input based on the arguments
e = input(f"Please provide input for e (num_epochs): ")
b = input(f"Please provide additional input for b (batch_size): ")
l = input(f"Please provide additional input for l (learning_rate): ")
# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = int(e)  # Number of times the entire dataset is passed through the model
batch_size = int(b)  # Number of samples per batch to be passed through the model
learning_rate = float(l)  # Step size for parameter updates

# MNIST dataset
# Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

# Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
# DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# DataLoader for the test dataset, used for evaluating the model
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Create an instance of the model & move it to the configured device (GPU/CPU)
model = utilities.ConvNet().to(device)

# Print the model for observation
# You can use the following libraries to observe the flowchart of the models similar to TensorFlow
# from torchview import draw_graph; from torchviz import make_dot
# see more at:
# https://github.com/mert-kurttutan/torchview
# https://github.com/szagoruyko/pytorchviz
print(model)

# Loss & optimizer
# CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
criterion = nn.CrossEntropyLoss()
# Adam optimizer with the specified learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model, loss_list, acc_train, acc_test = utilities.training(model, num_epochs,
                                                 train_loader, test_loader, criterion, optimizer)
# Make some plots!
utilities.plot(loss_list, acc_train, acc_test)