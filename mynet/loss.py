"""Loss functions."""

import numpy as np

from mynet.tensor import Tensor

class Loss:
    """Loss functions compares the real and predicted outputs."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """Calculate the loss."""
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """Calculate the gradient of the loss function."""
        raise NotImplementedError
    
class MSELoss(Loss):
    """Mean squared error loss."""
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """Calculate the loss."""
        return np.sum((predicted - actual) ** 2) / (2 * actual.size)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """Calculate the gradient of the loss function."""
        return (predicted - actual) /  actual.size