import numpy as np
from gym.envs.mujoco.walker2d import Walker2dEnv


from mbpo_pytorch.envs.virtual_env import BaseModelBasedEnv


class OriginalWalkerEnv(Walker2dEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        heights, angs = next_states[:, 0], next_states[:, 1]
        dones = np.logical_or(
            np.logical_or(heights >= 2.0, heights <= 0.8),
            np.abs(angs) >= 1.0
        )
        return None, dones
