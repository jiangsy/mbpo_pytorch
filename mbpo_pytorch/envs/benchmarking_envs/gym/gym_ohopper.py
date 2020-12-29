import numpy as np
from gym.envs.mujoco.hopper import HopperEnv


from mbpo_pytorch.envs.virtual_env import BaseModelBasedEnv


class OriginalHopperEnv(HopperEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        heights, angs = next_states[:, 0], next_states[:, 1]
        dones = np.logical_or(heights <= 0.7, abs(angs) >= 0.2)
        return None, dones
