"""Optimize the parameters of the neural net."""

from mynet.neuralnet import NeuralNet

class Optimizer:
    """Does the optization of the neural net."""
    def step(self, neural_net: NeuralNet)-> None:
        """Does one step of optimization."""
        raise NotImplementedError
    
class SGD(Optimizer):
    """Perform stochastic gradient descent."""
    def __init__(self, lr: float=0.01) -> None:
        """Lr is a learning rate, if is the proportional factor to the descent."""
        self.lr = lr

    def step(self, neural_net: NeuralNet) -> None:
        for param, grad in neural_net.param_and_grads():
            param -= self.lr * grad