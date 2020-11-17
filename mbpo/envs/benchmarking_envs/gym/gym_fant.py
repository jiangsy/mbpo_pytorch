import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mbrl.envs import BaseModelBasedEnv


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):

    def __init__(self, frame_skip=5):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/ant.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[13]
        reward_height = -3.0 * np.square(old_ob[0] - 0.57)

        # the alive bonus
        height = ob[0]
        done = (height > 1.0) or (height < 0.2)
        alive_reward = float(not done)

        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        # self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 13]
        reward_height = -3.0 * np.square(obs[:, 0] - 0.57)

        height = next_obs[:, 0]
        done = np.logical_or((height > 1.0), (height < 0.2))
        alive_reward = 1.0 - np.array(done, dtype=np.float)

        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return -reward

    def mb_step(self, states, actions, next_states):
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        height = next_states[:, 0]
        done = np.logical_or((height > 1.0), (height < 0.2))
        return rewards, done
