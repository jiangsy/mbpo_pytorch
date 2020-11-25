from __future__ import annotations
import abc
from abc import ABC
from typing import TYPE_CHECKING
from operator import itemgetter

import numpy as np
import torch
import gym

from mbpo_pytorch.thirdparty.base_vec_env import VecEnv
from mbpo_pytorch.storages import SimpleUniversalOffPolicyBuffer as Buffer

if TYPE_CHECKING:
    from mbpo_pytorch.models.dynamics import BaseDynamics


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


class VecVirtualEnv(VecEnv, ABC):
    def __init__(self, dynamics: BaseDynamics, env: BaseModelBasedEnv, num_envs: int, seed: int,
                 max_episode_steps=1000, use_auto_reset=True, use_predicted_reward=False):
        super(VecEnv, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.use_auto_reset = use_auto_reset
        self.use_predicted_reward = use_predicted_reward

        self.dynamics = dynamics
        self.device = next(self.dynamics.parameters()).device
        self.env = env
        self.env.seed(seed)

        self.action_lo_np = self.action_space.low
        self.action_hi_np = self.action_space.high

        self.action_lo_torch = torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)
        self.action_hi_torch = torch.tensor(self.action_space.high, dtype=torch.float32, device=self.device)

        if num_envs:
            self.elapsed_steps = np.zeros([self.num_envs], dtype=np.int32)
            self.episode_rewards = np.zeros([self.num_envs])
            self.states = np.zeros([self.num_envs, self.observation_space.shape[0]], dtype=np.float32)

    def _rescale_action_torch(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_lo_torch + (actions + 1.) * 0.5 * (self.action_hi_torch - self.action_lo_torch)

    def _rescale_action_np(self, actions: np.ndarray) -> np.ndarray:
        return self.action_lo_np + (actions + 1.) * 0.5 * (self.action_hi_np - self.action_lo_np)

    def step_with_states(self, states: torch.Tensor, actions: torch.Tensor, **kwargs):
        rescaled_actions = self._rescale_action_torch(actions)

        with torch.no_grad():
            if self.use_predicted_reward:
                next_states, rewards = itemgetter('next_states', 'rewards')(self.dynamics.predict(
                    states.to(self.device), actions.to(self.device), **kwargs))
                _, dones = self.env.mb_step(states.cpu().numpy(), rescaled_actions.cpu().numpy(),
                                            next_states.cpu().numpy())
            else:
                next_states, rewards = itemgetter('next_states')(self.dynamics.predict(
                    states.to(self.device), actions.to(self.device), **kwargs))
                rewards, dones = self.env.mb_step(states.cpu().numpy(), rescaled_actions.cpu().numpy(),
                                                  next_states.cpu().numpy())
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # no timeouts and auto_reset for step_with_states
        return next_states.clone(), rewards.clone(), dones, [{} for _ in range(len(states))]

    def step_async(self, actions: np.ndarray):
        self.actions = actions

    def step_wait(self):
        assert self.num_envs
        rescaled_actions = self._rescale_action_np(self.actions)
        self.elapsed_steps += 1

        with torch.no_grad():
            if self.use_predicted_reward:
                next_states, rewards = itemgetter('next_states', 'rewards')(self.dynamics.predict(
                    torch.tensor(self.states, device=self.device, dtype=torch.float32),
                    torch.tensor(self.actions, device=self.device, dtype=torch.float32)))
                next_states, rewards = next_states.cpu().numpy(), rewards.cpu().numpy()
                _, dones = self.env.mb_step(self.states, rescaled_actions, next_states)
            else:
                next_states, rewards = itemgetter('next_states')(self.dynamics.predict(
                    torch.tensor(self.states, device=self.device, dtype=torch.float32),
                    torch.tensor(self.actions, device=self.device, dtype=torch.float32)))
                next_states = next_states.cpu().numpy()
                rewards, dones = self.env.mb_step(self.states, rescaled_actions, next_states)

        self.episode_rewards += rewards
        self.states = next_states.copy()
        timeouts = self.elapsed_steps == self.max_episode_steps
        dones |= timeouts
        info_dicts = [{} for _ in range(self.num_envs)]
        for i, (done, timeout) in enumerate(zip(dones, timeouts)):
            if done:
                info = {'episode': {'r': self.episode_rewards[i], 'l': self.elapsed_steps[i]}}
                if timeout:
                    info.update({'TimeLimit.truncated': True})
                info_dicts[i] = info
            else:
                info_dicts[i] = {}
        if self.use_auto_reset:
            self.reset(np.argwhere(dones).squeeze(axis=-1))
        return self.states.copy(), rewards.copy(), dones.copy(), info_dicts

    # if indices = None, every env will be reset
    def reset(self, indices=None) -> np.ndarray:
        assert self.num_envs
        # have to distinguish [] and None
        indices = np.arange(self.num_envs) if indices is None else indices
        if np.size(indices) == 0:
            return np.array([])
        states = np.array([self.env.reset() for _ in indices])
        self.states[indices] = states
        self.elapsed_steps[indices] = 0
        self.episode_rewards[indices] = 0.
        return states.copy()

    # if indices = None, every env will be set
    def set_states(self, states: np.ndarray, indices=None):
        assert self.num_envs
        indices = indices or np.arange(self.num_envs)
        assert states.ndim == 2 and states.shape[0] == indices.shape[0]
        self.states[indices] = states.copy()
        # set_state should reset reward and length
        self.elapsed_steps[indices] = 0
        self.episode_rewards[indices] = 0.

    def close(self):
        pass

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, mode='human'):
        raise NotImplemented

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplemented

    def get_attr(self, attr_name, indices=None):
        raise NotImplemented

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplemented

