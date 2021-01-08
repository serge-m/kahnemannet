import random

from torch.utils.data import Sampler
from typing import List, Dict, Callable, Sequence


class RandIntSampler:
    def __init__(self, rs, min_len, max_len):
        self.rs = rs
        self.min_len, self.max_len = min_len, max_len

    def __call__(self):
        return self.rs.randint(self.min_len, self.max_len)


class ContinuousSequenceSampler(Sampler):
    def __init__(self, data_source,
                 sample_groups: Sequence[Sequence[int]],
                 subsequence_len_sampler: Callable[[], int],
                 random_state=None, seed=None):
        """
        Samples elements in sequences of random length. Each sequence contains elements from one class.

        @param data_source: parameter for upstream Sampler, used only for superclass initialization
        @param sample_groups:
        @param subsequence_len_sampler:
        @param random_state:
        @param seed:
        """
        super().__init__(data_source)
        if random_state is not None and seed is not None:
            raise ValueError("random_state and seed cannot be used at the same time")
        self.rs = random_state if random_state is not None else random.Random(seed)
        self.subsequence_len_sampler = subsequence_len_sampler
        self.iterators = {
            group_id: iter(self.rs.sample(indices, k=len(indices)))  # permutation, sample without replacement
            for group_id, indices in enumerate(sample_groups)
        }
        self.non_empty_groups = list(self.iterators.keys())
        self._len = sum([len(group) for group in sample_groups])

    def __iter__(self):
        while self.non_empty_groups:
            cl = self.rs.choice(self.non_empty_groups)
            it = self.iterators[cl]
            seq_len = self.subsequence_len_sampler()
            for i in range(seq_len):
                try:
                    yield next(it)
                except StopIteration:
                    self.non_empty_groups.remove(cl)
                    break

    def __len__(self):
        return self._len
