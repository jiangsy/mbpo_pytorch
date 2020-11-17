import abc
import gym
import numpy as np
import torch
from stable_baselines import logger

from mbpo.storages import SimpleUniversalOffPolicyBuffer as Buffer


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
        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        datatype = {'states': {'dims': [state_dim]}, 'next_states': {'dims': [state_dim]},
                    'actions': {'dims': [action_dim]}, 'rewards': {'dims': [1]}, 'masks': {'dims': [1]}}
        buffer = Buffer(n, datatype)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)

            states, actions, rewards, next_states = torch.tensor([state]), torch.tensor([action]), \
                                                    torch.tensor([reward]), torch.tensor([next_state])

            masks = torch.tensor([0.0] if done else [1.0], dtype=torch.float32)
            buffer.insert(states=states, actions=actions, rewards=rewards, masks=masks, next_states=next_states)
            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(buffer.states.numpy(), buffer.actions.numpy(), buffer.next_states.numpy())
        diff = (buffer.rewards.numpy() - rewards_) * buffer.masks.numpy()
        l_inf = np.abs(diff).max()
        logger.info('reward difference: %.6f', l_inf)

        assert np.allclose(dones_, buffer.masks), 'done model is inaccurate'
        assert l_inf < eps, 'reward model is inaccurate'

    def seed(self, seed: int = None):
        pass