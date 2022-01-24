import itertools
from typing import Sequence

import numpy as np
from torch.utils.data import Sampler


class BatchByLengthSampler(Sampler):

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        bucket_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(lengths)
        self.lengths = lengths
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rngesus = np.random.default_rng(seed)

    def __iter__(self):
        sorted_idxs = np.argsort(self.lengths)
        buckets = self._generate_buckets(sorted_idxs)
        num_batches = self.bucket_size // self.batch_size + 1
        for bucket in buckets:
            if self.shuffle:
                self.rngesus.shuffle(bucket)
            batches = [bucket[i:i+self.batch_size] for i in range(num_batches)]
            yield from itertools.chain.from_iterable(batches)

    def __len__(self):
        return len(self.lengths)

    def _generate_buckets(self, elements: Sequence[int]):
        num_buckets = len(elements) // self.bucket_size + 1
        buckets = [elements[i:i+self.bucket_size] for i in range(num_buckets)]
        return self.rngesus.permutation(buckets)
