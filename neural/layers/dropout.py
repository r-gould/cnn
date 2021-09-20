from .layer import Layer

class Dropout(Layer):
    def __init__(self, drop_prob):
        if not (0 < drop_prob < 1):
            raise ValueError("Dropout probability must be between 0 and 1.")

        self.drop_prob = drop_prob

    def _initialize(self, input_dim):
        self.output_dim = input_dim
        
    def _forward(self, X, training=True):
        mask = np.random.rand(*X.shape) < self.drop_prob
        
        if training:
            self.mask = mask

        O = (mask * X) / self.drop_prob
        return O

    def _backward(self, dO):
        dX = (self.mask * dO) / self.drop_prob
        return dX