import numpy as np

def linear(x):
    return x

def linear_prime(x):
    return 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return x * (x > 0)

def relu_prime(x):
    return x > 0

def softmax(z):
    z_exp = np.exp(z - np.max(z))
    return z_exp / np.sum(z_exp)

def softmax_prime(z):
    z = np.reshape(z, (-1, 1))
    jacobian = softmax(z) @ softmax(z).T
    jacobian = np.diag(softmax(z)) - jacobian
    return np.sum(jacobian, axis=1)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_prime(x):
    return 1 - tanh(x) ** 2

def zero_pad(X, pad):
    return np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)))