import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mbpo_pytorch.envs import BaseModelBasedEnv


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):

    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/swimmer.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)
        ob = self._get_obs()

        reward_ctrl = -0.0001 * np.square(action).sum()
        reward_run = old_ob[3]
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            # self.get_body_comvel("torso")[:1],
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                             self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.0001 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 3]
        reward = reward_run + reward_ctrl
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        """
        reward_ctrl = -0.0001 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 0]
        reward = reward_run + reward_ctrl
        return -reward
        """
        raise NotImplementedError
