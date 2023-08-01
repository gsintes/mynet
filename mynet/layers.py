"""Layers propagate input forward and gradients backgrounds."""

from typing import Dict, Callable

import numpy as np

from mynet.tensor import Tensor

class Layer:
    """A layer propagates input forward and gradients backgrounds."""
    def __init__(self) -> None:
        self.params : Dict[str, Tensor] = {}
        self.grads : Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """Propagates the input forward."""
        raise NotImplementedError
    
    def background(self, grad: Tensor) -> Tensor:
        """Back propagates the gradient."""
        raise NotImplementedError
    
class Linear(Layer):
    """Does a linear predictions of the input."""
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return  inputs @ self.params["w"] + self.params["b"]
    
    def background(self, grad: Tensor) -> Tensor:
        """if y = f(x) and x = w @ inputs + b:
             """
        self.grads["w"] =  self.inputs.T @ grad # input dim : batch * input_size; W dim : in*out
        self.grads["b"] = np.sum(grad, axis=0) #output dim : batch * out, grad : batch * out
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class ActivationLayer(Layer):
    """An activation layer applies directly a function element_wise to the inputs and backpropagate the gradients."""
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f 
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def background(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad
    
def tanh(x:Tensor) -> Tensor:
    """Just the hyperbolic tangent."""
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """Calculate the derivative of the hyperbolic tangent."""
    y = tanh(x)
    return 1 - y ** 2

class TanhActivation(ActivationLayer):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)