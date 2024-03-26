import numpy as np

def binary_cross_entropy(y_hat, y):
    epsilon = 1e-15  # para evitar divisiones por cero
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # para evitar log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def binary_cross_entropy_d(y_hat, y):
    epsilon = 1e-15  # para evitar divisiones por cero
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # para evitar log(0)
    return (y_hat - y) / (y_hat * (1 - y_hat))

def mse(y_hat, y):
  return np.mean((y_hat - y) ** 2)

def mse_d(y_hat, y):
  return (2 * (y_hat - y)) / len(y_hat)