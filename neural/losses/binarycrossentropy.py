import numpy as np

from .loss import Loss
from neural.constants import EPSILON

class BinaryCrossEntropy(Loss):
    def calculate_loss(self, y, yhat):
        return -np.sum(y * np.log(yhat+EPSILON) + (1-y) * np.log(1-yhat+EPSILON))

    def calculate_gradient(self, y, yhat):
        return -(y / (yhat+EPSILON) - (1-y) / (1-yhat+EPSILON))