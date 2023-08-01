"""Test the library trying to predicted the XOR."""

import numpy as np

from mynet.neuralnet import NeuralNet
from mynet.layers import Linear, TanhActivation
from mynet.train import train
from mynet.data import BatchIterator
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    ])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
]) # Encode XOR [1, 0] is not XOR, [0, 1] is XOR.

net = NeuralNet([
    Linear(2, 2)
])
train(net, inputs, targets, data_iterator=BatchIterator(3))

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y) 