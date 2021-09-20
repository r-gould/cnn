import sys
import numpy as np

from neural.utils import linear, linear_prime, sigmoid, sigmoid_prime, relu, relu_prime, softmax, softmax_prime
from .layer import Layer

sys.path.append("..")

class Activation(Layer):
    trainable = False
    
    str_to_funcs = {
        "sigmoid" : (sigmoid, sigmoid_prime),
        "relu" : (relu, relu_prime),
        "softmax" : (softmax, softmax_prime),
    }

    def __init__(self, activ_str):
        self.activ_str = activ_str
        (self.g, self.g_prime) = self.str_to_funcs[activ_str]

    def _initialize(self, input_dim):
        self.output_dim = input_dim
        
    def _forward(self, X, training=True):
        O = np.apply_along_axis(self.g, 1, X)

        if training:
            self.X = X
            self.O = O

        return O
    
    def _backward(self, dO):
        if self.activ_str == "softmax":
            y = -dO * self.O
            dX = self.O - y
            return dX

        dX = dO * np.apply_along_axis(self.g_prime, 1, self.X)
        return dX