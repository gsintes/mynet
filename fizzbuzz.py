"""Learn fizzbuzz."""

from typing import List

import numpy as np

from mynet.tensor import Tensor
from mynet.neuralnet import NeuralNet
from mynet.layers import Linear, TanhActivation, SigmoidActivation
from mynet.train import train

def encode_fizzbuzz(k: int) -> Tensor:
    """Encode fizzbuzz."""
    if k % 15 == 0:
        res = [0, 0, 0, 1]
    elif k % 5 == 0:
        res = [0, 0, 1, 0]
    elif k % 3 == 0:
        res = [0, 1, 0, 0]
    else:
        res = [1, 0, 0, 0]
    return np.array(res)

def binary_encoding(k: int, bit_number: int=10) -> List[int]:
    """Does the binary encryption"""
    return [k>>i & 1 for i in range(bit_number)]

inputs = np.array([binary_encoding(k) for k in range(101, 1024)])
targets = np.array([encode_fizzbuzz(k) for k in range(101, 1024)])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    TanhActivation(),
    Linear(input_size=50, output_size=4)
])

train(
    net,
    inputs,
    targets,
    nb_epochs=50000
)

error_count = 0
for x in range(1, 101):
    predicted = net.forward(binary_encoding(x))
    predicted_ind = np.argmax(predicted)
    actual_ind = np.argmax(encode_fizzbuzz(x))
    if predicted_ind != actual_ind:
        error_count += 1
    labels = [str(x), "Fizz", "Buzz", "Fizzbuzz"]
    print(x, labels[predicted_ind], labels[actual_ind])
print(error_count)