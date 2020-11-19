import abc
import gym
import numpy as np
import torch


from mbpo_pytorch.misc import logger
from mbpo_pytorch.storages import SimpleUniversalOffPolicyBuffer as Buffer


class BaseModelBasedEnv(gym.Env):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def seed(self, seed: int = None):
        pass