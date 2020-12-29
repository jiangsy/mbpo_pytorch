import numpy as np
from gym.envs.mujoco.ant import AntEnv


from mbpo_pytorch.envs.virtual_env import BaseModelBasedEnv


class OriginalAntEnv(AntEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        heights = next_states[:, 0]
        dones = np.logical_or((heights > 1.0), (heights < 0.2))
        return None, dones
