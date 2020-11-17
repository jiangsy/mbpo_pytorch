import abc
import gym
import numpy as np
import torch
from stable_baselines import logger

from mbpo.storages.off_policy_buffer import OffPolicyBuffer


class BaseBatchedEnv(gym.Env, abc.ABC):
    n_envs: int

    @abc.abstractmethod
    def step(self, actions):
        pass

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass

    def set_state(self, state):
        logger.warn('`set_state` is not implemented')


class BaseModelBasedEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def verify(self, n=2000, eps=1e-4):
        buffer = OffPolicyBuffer(n, self.observation_space.shape, 1, self.action_space)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)

            mask = torch.tensor([0.0] if done else [1.0], dtype=torch.float32)

            buffer.insert(torch.tensor(state), torch.tensor(action), torch.tensor(reward),
                          torch.tensor(next_state), torch.tensor(mask))

            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(buffer.states.numpy(), buffer.actions.numpy(), buffer.next_states.numpy())
        diff = (buffer.rewards.numpy() - rewards_[:, np.newaxis]) * buffer.masks.numpy()
        l_inf = np.abs(diff).max()
        logger.info('reward difference: %.6f', l_inf)

        assert np.allclose(dones_, buffer.masks), 'reward model is inaccurate'
        assert l_inf < eps, 'done model is inaccurate'

    def seed(self, seed: int = None):
        pass