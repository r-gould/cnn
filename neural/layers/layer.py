class Layer:
    def _initialize(self, input_dim):
        raise NotImplementedError()

    def _forward(self, X, training=True):
        raise NotImplementedError()

    def _backward(self, dO):
        raise NotImplementedError()