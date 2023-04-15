# Neural-Net-in-C

Linear algebra and neural network library writing multi-layer perceptrons in C. Extends [Logistic Regression in C](https://github.com/AlephEleven/logistic-regression-in-C).

## Features

### Matrices

* Pseudo-random generator for gaussian distributed numbers
* Matrix object for handling data, and shape+size properties
* Tools for applying linear algebra techniques such as matrix multplication, transpose, addition, elementwise functions
* Warnings for bad matrix multiplication/elementwise multiplication

### Neural Network

* Macro-based objects for creating linear layers/activation/loss parameters efficiently
* Sigmoid Activation and MeanSquaredLoss
* Customizable forward pass function
* Generalized Training loop + backpropagation algorithm
* One-hot encoding accuracy measure
* Dataset object for holding training data


## Example: Iris Dataset with one-hot encoding


![alt text](https://github.com/AlephEleven/Neural-Net-in-C/blob/main/results.PNG?raw=true)
