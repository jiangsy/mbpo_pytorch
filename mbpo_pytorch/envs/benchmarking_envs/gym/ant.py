import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mbpo_pytorch.envs import BaseModelBasedEnv


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):

    def __init__(self, frame_skip=5):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/ant.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def step(self, action: np.ndarray):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[13]
        reward_height = -3.0 * np.square(old_ob[0] - 0.57)
        reward = reward_run + reward_ctrl + reward_height + 1.0
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
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
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
        reward = reward_run + reward_ctrl + reward_height + 1.0
        return -reward

