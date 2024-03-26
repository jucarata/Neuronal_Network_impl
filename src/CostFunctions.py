import numpy as np

def binary_cross_entropy(y_hat, y):
  return -1 / y.shape[1] * np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)), axis=1, keepdims=True)

def binary_cross_entropy_d(y_hat, y):
  return 1 / y.shape[1] * (y_hat - y) / (y_hat * (1 - y_hat))

def mse(y_hat, y):
  return np.mean((y_hat - y) ** 2)

def mse_d(y_hat, y):
  return (2 * (y_hat - y)) / len(y_hat)