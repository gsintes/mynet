"""Enable the data grouped in batches."""

from typing import Any, NamedTuple, Iterator

import numpy as np

from mynet.tensor import Tensor

Batch = NamedTuple("Batch",[("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError
    
class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool=True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        assert len(inputs) == len(targets)
        indexes = np.arange(0, len(inputs))
        if self.shuffle:
            np.random.shuffle(indexes)
        starts = np.arange(0, len(inputs), self.batch_size)
        for start in starts:
            end = start + self.batch_size
            batch_indexes = indexes[start:end]
            batch_inputs = inputs[batch_indexes]
            batch_targets = targets[batch_indexes]
            yield Batch(batch_inputs, batch_targets)