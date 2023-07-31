"""Layers propagate input forward and gradients backgrounds."""

import numpy as np

from mynet.tensor import Tensor

class Layer:
    """A layer propagates input forward and gradients backgrounds."""
    def __init__(self) -> None:
        pass

    def forward(self, input: Tensor) -> Tensor:
        """Propagates the input forward."""
        raise NotImplementedError
    
    def background(self, grad: Tensor) -> Tensor:
        """Back propagates the gradient."""
        raise NotImplementedError