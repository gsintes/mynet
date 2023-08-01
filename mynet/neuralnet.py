"""A neural network is a sequence of layers."""

from typing import Sequence, Iterator, Tuple

from mynet.layers import Layer
from mynet.loss import Loss, MSELoss
from mynet.optimizers import Optimizer, SGD
from mynet.tensor import Tensor

class NeuralNet:
    def __init__(self, layers : Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        """Push the input forward."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        """Backpropagate the gradient through the layers."""
        for layer in reversed(self.layers):
            grad = layer.background(grad)
        return grad
    
    def param_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Return an iterator over the parameters and there gradients for all layers."""
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def train(self,
              loss: Loss = MSELoss(),
              optimizer: Optimizer= SGD()
              )-> None:
        """Train the network with training data."""
        pass