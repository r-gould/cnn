import numpy as np
from .layer import Layer

class Dense(Layer):
    trainable = True

    def __init__(self, units):
        self.units = units

    def _initialize(self, input_dim):
        prev_units, = input_dim
        self.output_dim = (self.units,)

        self.W = np.random.randn(self.units, prev_units) * 0.01
        self.b = np.zeros((1, self.units))

    def _forward(self, X, training=True):
        if training:
            self.X = X

        O = X @ self.W.T + self.b
        return O

    def _backward(self, dO):
        self.dW = dO.T @ self.X
        self.db = np.sum(dO, axis=0, keepdims=True)
        dX = dO @ self.W
        return dX