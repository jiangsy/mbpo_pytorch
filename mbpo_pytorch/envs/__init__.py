import abc
from operator import itemgetter

import gym
import numpy as np
import torch


from mbpo_pytorch.misc import logger
from mbpo_pytorch.storages import SimpleUniversalOffPolicyBuffer as Buffer


class BaseModelBasedEnv(gym.Env):
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

        states, actions, rewards, masks, next_states = \
            itemgetter('states', 'actions', 'rewards', 'masks', 'mask_states')(next(buffer.get_batch_generator(None)))

        rewards_gt, dones_gt = self.mb_step(states.numpy(), actions.numpy(), next_states.numpy())
        diff = (rewards.numpy() - rewards_gt) * masks.numpy()
        l_inf = np.abs(diff).max()
        logger.info('reward difference: {:.4f}'.format(l_inf))

        assert np.allclose(1.0 - dones_gt, masks.numpy()), 'done model is inaccurate'
        assert l_inf < eps, 'reward model is inaccurate'

    def seed(self, seed: int = None):
        pass
