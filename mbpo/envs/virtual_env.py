from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from operator import itemgetter

import gym
import numpy as np
import torch

from mbrl.thirdparty.base_vec_env import VecEnv
if TYPE_CHECKING:
    from mbrl.models.dynamics import BaseDynamics
    from mbrl.envs import BaseModelBasedEnv


class VecVirtualEnv(VecEnv):
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

        self.action_lo = torch.tensor(self.action_space.low, dtype=torch.float32, device=self.device)
        self.action_hi = torch.tensor(self.action_space.high, dtype=torch.float32, device=self.device)

        if num_envs:
            self.elapsed_steps = np.zeros([self.num_envs], dtype=np.int32)
            self.episode_rewards = np.zeros([self.num_envs])
            self.states = np.zeros([self.num_envs, self.observation_space.shape[0]], dtype=np.float32)

    def _rescale_action(self, actions: torch.Tensor):
        return self.action_lo + (actions + 1.) * 0.5 * (self.action_hi - self.action_lo)

    def step_with_states(self, states, actions, **kwargs):
        rescaled_actions = self._rescale_action(actions)

        with torch.no_grad():
            if self.use_predicted_reward:
                next_states, rewards = itemgetter('next_states', 'rewards')(self.dynamics.predict(
                    states.to(self.device), actions.to(self.device), **kwargs))
                _, dones = self.env.mb_step(states.cpu().numpy(), rescaled_actions.cpu().numpy(),
                                            next_states.cpu().numpy())
            else:
                next_states, rewards = itemgetter('next_states')(self.dynamics.predict(
                    states.to(self.device), actions.to(self.device), **kwargs))
                rewards, dones = self.env.mb_step(states, rescaled_actions, next_states)
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        # no timeouts and reset for step_with_states
        return next_states.clone(), rewards.clone(), dones, [{} for _ in range(len(states))]

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        assert self.num_envs
        rescaled_actions = self._rescale_action(self.actions)
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

