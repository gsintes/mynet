"""Train the neural net."""

from mynet.loss import Loss, MSELoss
from mynet.optimizers import Optimizer, SGD
from mynet.data import DataIterator, BatchIterator
from mynet.tensor import Tensor
from mynet.neuralnet import NeuralNet

def train(neural_net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          nb_epochs: int = 5000,
          data_iterator: DataIterator = BatchIterator(),
          loss: Loss = MSELoss(),
          optimizer: Optimizer= SGD()
          )-> None:
    """Train the network with training data."""
    for epoch in range(nb_epochs):
        epoch_loss = 0.

        for data in data_iterator(inputs, targets):
            predicted = neural_net.forward(data.inputs)
            epoch_loss += loss.loss(predicted, data.targets)
            grad = loss.grad(predicted, data.targets)
            neural_net.backward(grad)
            optimizer.step(neural_net)
        print(epoch, epoch_loss)