import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

from mbrl.envs import BaseModelBasedEnv


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat])

    def step(self, a):
        data = self.sim.data
        action = a
        if getattr(self, 'action_space', None):
            action = np.clip(a, self.action_space.low,
                             self.action_space.high)

        # reward
        alive_bonus = 5.0
        lin_vel_cost = 0.25 / 0.015 * data.qvel.flat[0]
        quad_ctrl_cost = 0.1 * np.square(action).sum()
        quad_impact_cost = 0.0

        self.do_simulation(action, self.frame_skip)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = 0.25 / 0.015 * obs[:, 22]

        quad_impact_cost = 0.0

        height = next_obs[:, 0]
        done = np.logical_or((height > 2.0), (height < 1.0))
        alive_reward = 5 * (1.0 - np.array(done, dtype=np.float))

        reward = reward_run + reward_ctrl + (-quad_impact_cost) + alive_reward
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        raise NotImplementedError

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        height = next_states[:, 0]
        done = np.logical_or((height > 2.0), (height < 1.0))
        return rewards, done

    def verify(self):
        pass
