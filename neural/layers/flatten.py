import numpy as np

from .layer import Layer

class Flatten(Layer):
    trainable = False

    def _initialize(self, input_dim):
        (X_r, X_c, X_ch) = input_dim
        self.output_dim = (X_r * X_c * X_ch,)

    def _forward(self, X, training=True):
        if training:
            self.original_shape = X.shape

        m = X.shape[0]
        O = np.reshape(X, (m, -1))
        return O

    def _backward(self, dO):
        dX = np.reshape(dO, self.original_shape)
        return dX