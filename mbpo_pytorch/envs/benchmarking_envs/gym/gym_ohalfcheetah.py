import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


from mbpo_pytorch.envs.virtual_env import BaseModelBasedEnv


class OriginalHalfCheetahEnv(HalfCheetahEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        return None, np.zeros([states.shape[0], 1], dtype=np.bool)
