import numpy as np
from gym.envs.mujoco.humanoid import HumanoidEnv


from mbpo_pytorch.envs.virtual_env import BaseModelBasedEnv


class OriginalHumanoidEnv(HumanoidEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        heights = next_states[:, 0]
        dones = np.logical_or((heights > 2.0), (heights < 1.0))
        return None, dones
