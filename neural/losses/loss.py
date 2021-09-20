class Loss:
    def calculate_loss(self, y, yhat):
        raise NotImplementedError()

    def calculate_gradient(self, y, yhat):
        raise NotImplementedError()