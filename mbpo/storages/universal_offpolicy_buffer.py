from __future__ import annotations
from typing import Dict, Generator, Optional, Union, TYPE_CHECKING
from functools import reduce
from operator import add, itemgetter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler, BatchSampler, SubsetRandomSampler

from mbrl.misc import logger
if TYPE_CHECKING:
    from mbrl.storages.universal_onpolicy_buffer import UniversalOnPolicyBuffer
    from mbrl.storages.universal_traj_buffer import UniversalTrajBuffer


# noinspection DuplicatedCode
class SimpleUniversalOffPolicyBuffer:
    def __init__(self, buffer_size: int,
                 entry_dict: Dict[str, dict], **kwargs):
        self.entry_infos = entry_dict.copy()
        self.entries = self.entry_infos.keys()
        self.entry_num_classes = {}
        self.entry_sizes = {}
        self.entry_indices = {}
        self.device = torch.device('cpu')
        self.buffer_size = buffer_size
        self.size = 0
        self.index = 0

        for name, properties in entry_dict.items():
            dims = properties.get('dims')
            dtype = torch.float32
            if isinstance(dims, int):
                self.entry_num_classes[name] = dims
                dims = [1]
                dtype = torch.int32
            self.__dict__[name] = torch.zeros(buffer_size, *dims, dtype=dtype)

        for name, value in kwargs.items():
            setattr(self, name, value)

    def to(self, device, inplace=False):
        self.device = device
        if inplace:
            for entry in self.entry_infos.keys():
                self.__dict__[entry] = self.__dict__[entry].to(device)

    def insert(self, **kwargs):
        lengths = []
        for (name, value) in kwargs.items():
            lengths.append(value.shape[0])
            self._insert_sequence(name, value, self.index)
        assert len(set(lengths)) == 1
        self.size = min(self.size + lengths[0], self.buffer_size)
        self.index = (self.index + lengths[0]) % self.buffer_size

    def clear(self):
        self.index = self.size = 0

    def _insert_sequence(self, name, values, index):
        length = values.shape[0]
        if index + length <= self.buffer_size:
            self.__dict__[name][index: index + length].copy_(values)
        else:
            self._insert_sequence(name, values[:self.buffer_size - index], index)
            self._insert_sequence(name, values[self.buffer_size - index:], 0)

    def add_buffer(self, buffer: Union[UniversalOnPolicyBuffer, SimpleUniversalOffPolicyBuffer, UniversalTrajBuffer]):
        if isinstance(buffer, UniversalOnPolicyBuffer):
            self._add_onpolicy_buffer(buffer)
        elif isinstance(buffer, SimpleUniversalOffPolicyBuffer):
            self._add_offpolicy_buffer(buffer)
        elif isinstance(buffer, UniversalTrajBuffer):
            self._add_traj_buffer(buffer)

    def save(self, save_path):
        save_result = {'index': self.index, 'size': self.size, 'entry_infos': self.entry_infos}
        for name in set(self.entry_infos.keys()):
            save_result[name] = self.__dict__[name].cpu().clone()
        torch.save(save_result, save_path)

    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.index, self.size, self.entry_infos = itemgetter('index', 'size', 'entry_infos')(state_dict)
        buffer_size = []
        for name in self.entry_infos:
            self.__dict__[name] = state_dict[name]
            buffer_size.append(state_dict[name].shape[0])
        assert len(set(buffer_size)) == 1
        self.resize(buffer_size[0])

    def get_batch_generator_inf(self, batch_size: Optional[int], ranges=None) -> Generator:
        ranges = range(self.size) if ranges is None else ranges
        batch_size = batch_size or len(ranges)
        while True:
            indices = np.fromiter(RandomSampler(ranges, replacement=True, num_samples=batch_size), np.int32)
            batch = {}
            for name in self.entry_infos.keys():
                batch[name] = self.__dict__[name].view(-1, *self.__dict__[name].shape[1:])[indices].clone().\
                    to(self.device)
                if name in self.entry_num_classes and getattr(self, 'use_onehot_output', False):
                    batch[name] = F.one_hot(batch[name].long(), num_classes=self.entry_num_classes[name]).\
                        squeeze(-2).float()
            yield batch

    def get_batch_generator(self, batch_size: Optional[int], ranges=None) -> Generator:
        ranges = range(self.size) if ranges is None else ranges
        batch_size = batch_size or len(ranges)
        sampler = BatchSampler(SubsetRandomSampler(range(self.size)), batch_size, drop_last=True)
        for indices in sampler:
            batch = {}
            for name in self.entry_infos.keys():
                batch[name] = self.__dict__[name].view(-1, *self.__dict__[name].shape[1:])[indices].clone().to(self.device)
                if name in self.entry_num_classes and getattr(self, 'use_onehot_output', False):
                    batch[name] = F.one_hot(batch[name].long(), num_classes=self.entry_num_classes[name]).\
                        squeeze(-2).float()
            yield batch

    def resize(self, buffer_size):
        if buffer_size == self.buffer_size:
            return
        if buffer_size < self.buffer_size:
            logger.info('Buffer resize from {} to {}'.format(self.buffer_size, buffer_size))
            for name in self.entry_infos:
                self.__dict__[name] = self.__dict__[name][:buffer_size]
            self.size = min(self.size, buffer_size)
            self.buffer_size = buffer_size
            if self.index >= buffer_size:
                self.index = 0
            return
        if buffer_size > self.buffer_size:
            logger.info('Buffer resize from {} to {}'.format(self.buffer_size, buffer_size))
            for name in self.entry_infos:
                value = self.__dict__[name]
                self.__dict__[name] = torch.cat([value, torch.zeros(buffer_size - self.buffer_size, *value.shape[1:],
                                                                    dtype=value.dtype, device=value.device)],
                                                dim=0)
            self.buffer_size = buffer_size
            if self.index < self.size:
                self.index = self.size
            return

    def _add_offpolicy_buffer(self, buffer: SimpleUniversalOffPolicyBuffer):
        length = buffer.size
        for name in self.entry_infos.keys():
            assert self.__dict__[name].shape[1:] == buffer.__dict__[name].shape[1:]
            self._insert_sequence(name, buffer.__dict__[name][:length], self.index)
        self.size = min(self.size + length, self.buffer_size)
        self.index = (self.index + length) % self.buffer_size

    def _add_onpolicy_buffer(self, buffer: UniversalOnPolicyBuffer):
        valid_indices = buffer.calculate_valid_indices(0)
        length = len(valid_indices)
        for name in self.entry_infos.keys():
            assert self.__dict__[name].shape[1:] == buffer.__dict__[name].shape[2:]
            self._insert_sequence(name, buffer.__dict__[name].view(-1, *self.__dict__[name].shape[1:])
            [valid_indices], self.index)
        self.size = min(self.size + length, self.buffer_size)
        self.index = (self.index + length) % self.buffer_size

    def _add_traj_buffer(self, buffer: UniversalTrajBuffer):
        for i in buffer.size:
            length = buffer.lengths[i]
            for entry in self.entry_infos.keys():
                self._insert_sequence(entry, buffer.__dict__[entry][i][:buffer.lengths[i]], self.index)
            self.size = min(self.size + length, self.buffer_size)
            self.index = (self.index + length) % self.buffer_size


# noinspection DuplicatedCode
class UniversalOffPolicyBuffer:
    def __init__(self, buffer_size: int,
                 entry_dict: Dict[str, dict], **kwargs):
        self.entry_dict = entry_dict.copy()
        self.entries = self.entry_dict.keys()
        self.entry_num_classes = {}
        self.entry_sizes = {}
        self.entry_indices = {}
        self.buffer_size = buffer_size
        self.device = torch.device('cpu')

        self.keep_traj = True
        self.size = 0
        self.index = 0

        for name, properties in entry_dict.items():
            dims = properties.get('dims')
            dtype = torch.float32
            if isinstance(dims, int):
                self.entry_num_classes[name] = dims
                dims = [1]
                dtype = torch.int32
            self.__dict__[name] = torch.zeros(buffer_size, *dims, dtype=dtype)
            self.entry_sizes[name] = 0
            self.entry_indices[name] = 0

        for name, value in kwargs.items():
            setattr(self, name, value)

    def to(self, device, inplace=False):
        self.device = device
        if inplace:
            for entry in self.entry_dict.keys():
                self.__dict__[entry] = self.__dict__[entry].to(device)

    def insert(self):
        self.keep_traj = False
        pass

    def clear(self):
        for name in self.entry_dict.keys():
            self.entry_sizes[name] = 0
            self.entry_indices[name] = 0

    # check all non-empty entries should have a same size and index
    def _check(self):
        sizes = self.entry_sizes.values()
        indices = self.entry_indices.values()
        sizes = set(filter(lambda x: x != 0, sizes))
        indices = set(filter(lambda x: x != 0, indices))
        if not len(sizes) == len(indices) == 1:
            logger.error('Inconsistent content sizes or indices: \n'
                         '\t{}\n'
                         '\t{}\n'.format(self.entry_sizes, self.entry_indices))
        self.size = list(sizes)[0]

    def _insert(self, name, values):
        length = values.shape[0]
        if self.entry_indices[name] + length < self.buffer_size:
            self.__dict__[name][self.entry_indices[name]: self.entry_indices[name] + length].copy_(values)
            self.entry_sizes[name] = min(self.buffer_size + length, self.entry_sizes[name])
            self.entry_indices[name] = (self.entry_indices[name] + length) % self.buffer_size
        else:
            self._insert(name, values[:self.buffer_size-self.entry_indices[name]])
            self._insert(name, values[self.buffer_size-self.entry_indices[name]:])

    def add_onpolicy_buffer(self, onpolicy_buffer: UniversalOnPolicyBuffer):
        valid_indices = onpolicy_buffer.calculate_valid_indices(0)
        entries = set(self.entry_dict.keys()).intersection(onpolicy_buffer.entry_dict.keys())
        for name in entries:
            assert self.__dict__[name].shape[1:] == onpolicy_buffer.__dict__[name].shape[2:]
            self._insert(name, onpolicy_buffer.__dict__[name].view(-1, *self.__dict__[name].shape[1:])[valid_indices])
        self.size += len(valid_indices)
        self._check()

    def add_traj_buffer(self, traj_buffer: UniversalTrajBuffer):
        entries = set(self.entry_dict.keys()).intersection(traj_buffer.entry_dict.keys())
        for i in traj_buffer.size:
            for entry in entries:
                self._insert(entry, traj_buffer.__dict__[entry][i][:traj_buffer.lengths[i]])
        self._check()

    def add_entries(self, entry_dict: Dict[str, dict], **kwargs):
        self.entry_dict.update(entry_dict)
        for entry in entry_dict.keys():
            self.__dict__[entry] = kwargs[entry].clone()
        self._check()

    def save(self, save_path):
        self._check()
        empty_entries = set([entry for entry in self.entry_sizes.keys() if self.entry_sizes[entry] == 0])
        save_result = {}
        for entry in set(self.entry_dict.keys()).difference(empty_entries):
            save_result[entry] = self.__dict__[entry].cpu().clone()
        torch.save(save_result, save_path)

    def load(self, load_path):
        pass

    # noinspection DuplicatedCode
    def get_batch_generator(self, batch_size: Optional[int], num_steps=1, device=None) -> Generator:
        empty_entries = set([entry for entry in self.entry_sizes.keys() if self.entry_sizes[entry] == 0])
        device = device or self.device
        while True:
            indices = list(RandomSampler(range(self.size - num_steps + 1), replacement=True, num_samples=batch_size))
            batch = {}
            indices = np.array(indices)
            for entry in set(self.entry_dict.keys()).difference(empty_entries):
                res = []
                for i in range(num_steps):
                    res.append(
                        self.__dict__[entry].view([-1] + list(self.__dict__[entry].shape[2:]))[indices + i].clone())
                # [batch, num_steps, _]
                batch[entry] = torch.stack(res).transpose(0, 1).to(device)
                if entry in self.entry_num_classes:
                    # nn.functional.one_hot requires torch.int64
                    batch[entry] = F.one_hot(batch[entry].long(), num_classes=self.entry_num_classes[entry]).\
                        squeeze(-2).float()
            yield batch


# noinspection DuplicatedCode
class UniversalMultiEnvOffPolicyBuffer:
    def __init__(self, buffer_size: int, num_envs,
                 entry_infos: Dict[str, dict], **kwargs):
        self.entry_infos = entry_infos.copy()
        self.entries = self.entry_infos.keys()
        self.entry_num_classes = {}
        self.entry_sizes = {}
        self.entry_indices = {}
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = torch.device('cpu')

        self.size = 0
        self.index = 0

        for name, properties in entry_infos.items():
            dims = properties.get('dims')
            dtype = torch.float32
            if isinstance(dims, int):
                self.entry_num_classes[name] = dims
                dims = [1]
                dtype = torch.int32
            self.__dict__[name] = torch.zeros(buffer_size, num_envs, *dims, dtype=dtype)

        for name, value in kwargs.items():
            setattr(self, name, value)

    def to(self, device, inplace=False):
        self.device = device
        if inplace:
            for entry in self.entry_infos.keys():
                self.__dict__[entry] = self.__dict__[entry].to(device)

    def clear(self):
        self.size = self.index = 0

    def insert(self, **kwargs):
        for (name, value) in kwargs:
            if not value:
                continue
            self.__dict__[name][self.entry_sizes[name]].copy_(value)

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def _insert(self, name, values, index):
        length = values.shape[0]
        if index + length < self.buffer_size:
            self.__dict__[name][index: index + length].copy_(values)
        else:
            self._insert(name, values[:self.buffer_size - index], index)
            self._insert(name, values[self.buffer_size - self.entry_indices[name]:], 0)

    def add_buffer(self, buffer: Union[UniversalOnPolicyBuffer, UniversalMultiEnvOffPolicyBuffer]):
        length = buffer.calculate_size()
        for name in self.entry_infos.keys():
            assert self.__dict__[name].shape[1:] == buffer.__dict__[name].shape[1:]
            self._insert(name, buffer.__dict__[name][:length], self.index)
        self.size = min(self.size + length, self.buffer_size)
        self.index = (self.index + length) % self.buffer_size

    def calculate_size(self):
        return self.size

    def save(self, save_path):
        save_dict = {'size': self.size, 'index': self.index, 'entry_dict': self.entry_infos}
        for entry in self.entry_infos.keys():
            save_dict.update({entry: self.__dict__[entry].cpu()})
        torch.save(save_dict, save_path)

    def load(self, load_path):
        load_dict = torch.load(load_path)
        self.size = load_dict['size']
        self.index = load_dict['index']
        self.entry_infos = load_dict['entry_dict']
        for entry in self.entry_infos.keys():
            self.__dict__[entry] = load_dict[entry]

    # noinspection DuplicatedCode
    def get_batch_generator_inf(self, batch_size: Optional[int], num_steps=1, ) -> Generator:
        valid_indices = reduce(add, [list(range(i * self.buffer_size, i * self.buffer_size + self.size - num_steps + 1))
                                     for i in self.num_envs])
        batch_size = batch_size or len(valid_indices)
        while True:
            indices = np.fromiter(RandomSampler(valid_indices, replacement=True, num_samples=batch_size), np.int32)
            batch = {}
            for entry in self.entry_infos.keys():
                res = []
                for i in range(num_steps):
                    res.append(
                        self.__dict__[entry].view([-1] + list(self.__dict__[entry].shape[2:]))[indices + i].clone())
                # [batch, num_steps, _]
                batch[entry] = torch.stack(res).transpose(0, 1).to(self.device)
                if entry in self.entry_num_classes:
                    batch[entry] = F.one_hot(batch[entry].long(), num_classes=self.entry_num_classes[entry]).\
                        squeeze(-2).float()
            yield batch


