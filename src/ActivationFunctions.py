import numpy as np

def sigmoid(x):
    return 1/(1 + np.e ** (-x))

def sigmoid_d(x):
    return np.exp(-x) / np.square(1 + np.exp(-x))

def sigmoid_d(x):
    return np.e ** (-x) / (1 + np.e ** (-x)) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(int)