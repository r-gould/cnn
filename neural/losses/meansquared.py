import numpy as np

from .loss import Loss

class MeanSquared(Loss):
    def calculate_loss(self, y, yhat):
        return np.sum(np.square(y-yhat))

    def calculate_gradient(self, y, yhat):
        return 2 * (yhat - y)