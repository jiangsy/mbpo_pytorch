from abc import ABC

import torch
import torch.nn as nn
from typing import List

from mbpo_pytorch.thirdparty.running_mean_std import RunningMeanStd


class RunningNormalizer(nn.Module, ABC):
    def __init__(self, shape: List[int], eps=1e-8, verbose=0):  # batch_size x ...
        super().__init__()

        self.shape = shape
        self.verbose = verbose

        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.eps = eps
        self.count = 1e-4

    def forward(self, x: torch.Tensor, inverse=False):
        if inverse:
            return x * torch.sqrt(self.var) + self.mean
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.var = self.var.to(*args, **kwargs)

    def update(self, samples: torch.Tensor):
        sample_count = samples.shape[0]
        sample_mean = samples.mean(dim=0)
        sample_var = samples.var(dim=0, unbiased=False)
        delta = sample_mean - self.mean
        total_count = self.count + sample_count

        new_mean = self.mean + delta * sample_count / total_count
        m_a = self.var * self.count
        m_b = sample_var * sample_count
        m_2 = m_a + m_b + delta * delta * self.count * sample_count / (self.count + sample_count)
        new_var = m_2 / (self.count + sample_count)

        new_count = sample_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def state_dict(self, *args, **kwargs):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']

    def get_rms(self):
        rms = RunningMeanStd(self.shape)
        rms.count = self.count
        rms.mean = self.mean.cpu().numpy()
        rms.var = self.var.cpu().numpy()
        return rms


class BatchNormalizer(nn.Module, ABC):
    def __init__(self, shape: List[int], eps=1e-8, verbose=0):
        super().__init__()

        self.shape = shape
        self.verbose = verbose

        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.std = torch.ones(shape, dtype=torch.float32)
        self.eps = eps

    def forward(self, x: torch.Tensor, inverse=False):
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / (torch.clamp(self.std, min=self.eps))

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)

    # noinspection DuplicatedCode
    # samples in [batch_size, ...]
    def update(self, samples: torch.Tensor):
        self.mean = torch.mean(samples, dim=0)
        self.std = torch.std(samples, dim=0)

    # noinspection PyMethodOverriding
    def state_dict(self, *args, **kwargs):
        return {'mean': self.mean, 'std': self.std}

    # noinspection PyMethodOverriding
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Normalizers(nn.Module, ABC):
    def __init__(self, dim_action: int, dim_state: int, verbose=0):
        super().__init__()
        # action_normalizer is not used
        self.action_normalizer = RunningNormalizer([dim_action], verbose=verbose)
        self.state_normalizer = RunningNormalizer([dim_state], verbose=verbose)
        self.diff_normalizer = RunningNormalizer([dim_state], verbose=verbose)

    def forward(self):
        raise NotImplemented

    def to(self, *args, **kwargs):
        self.action_normalizer.to(*args, **kwargs)
        self.state_normalizer.to(*args, **kwargs)
        self.diff_normalizer.to(*args, **kwargs)

    # noinspection PyMethodOverriding
    def state_dict(self, *args, **kwargs):
        return {'action_normalizer': self.action_normalizer.state_dict(),
                'state_normalizer': self.state_normalizer.state_dict(),
                'diff_normalizer': self.diff_normalizer.state_dict()}

    def load_state_dict(self, state_dict, stict):
        self.action_normalizer.load_state_dict(state_dict['action_normalizer'])
        self.state_normalizer.load_state_dict(state_dict['state_normalizer'])
        self.diff_normalizer.load_state_dict(state_dict['diff_normalizer'])
