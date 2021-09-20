import numpy as np

from .loss import Loss
from neural.constants import EPSILON

class CrossEntropy(Loss):
    def calculate_loss(self, y, yhat):
        return -np.sum(y * np.log(yhat+EPSILON))

    def calculate_gradient(self, y, yhat):
        return -y / (yhat+EPSILON)