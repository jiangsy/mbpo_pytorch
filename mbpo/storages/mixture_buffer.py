import numpy as np
import torch

from mbpo.misc.utils import merge_dicts
from mbpo.misc import logger


class MixtureBuffer:
    def __init__(self, buffers, weights, verbose=0):
        self.buffers = buffers
        self.weights = np.array(weights)
        self.verbose = verbose

    def get_batch_generator_inf(self, batch_size, **kwargs):
        batch_sizes = (batch_size * self.weights).astype(np.int)
        if self.verbose:
            logger.log('[Buffer Mixing] Max error {}'.format(np.max(np.abs(batch_sizes / batch_size - self.weights))))
        rand_index = np.random.randint(len(batch_sizes))
        batch_sizes[rand_index] = batch_size - np.delete(batch_sizes, rand_index).sum()
        inf_gens = [buffer.get_batch_generator_inf(int(batch_size_), **kwargs)
                    for buffer, batch_size_ in zip(self.buffers, batch_sizes)]
        while True:
            buffer_samples = list(map(lambda gen: next(gen), inf_gens))
            yield merge_dicts(buffer_samples, lambda x: torch.cat(x, dim=0))
