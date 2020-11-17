import numpy as np

from mbrl.envs.benchmarking_envs.gym.half_cheetah import HalfCheetahEnv
from mbrl.envs.benchmarking_envs.gym.walker2d import Walker2dEnv
from mbrl.envs.benchmarking_envs.gym.ant import AntEnv
from mbrl.envs.benchmarking_envs.gym.hopper import HopperEnv
from mbrl.envs.benchmarking_envs.gym.swimmer import SwimmerEnv
from mbrl.envs.benchmarking_envs.gym.reacher import ReacherEnv
from mbrl.envs.benchmarking_envs.gym.pendulum import PendulumEnv
from mbrl.envs.benchmarking_envs.gym.inverted_pendulum import InvertedPendulumEnv
from mbrl.envs.benchmarking_envs.gym.acrobot import AcrobotEnv
from mbrl.envs.benchmarking_envs.gym.cartpole import CartPoleEnv
from mbrl.envs.benchmarking_envs.gym.mountain_car import Continuous_MountainCarEnv
from mbrl.envs.benchmarking_envs.gym import gym_fswimmer
from mbrl.envs.benchmarking_envs.gym import gym_fwalker2d
from mbrl.envs.benchmarking_envs.gym import gym_fhopper
from mbrl.envs.benchmarking_envs.gym import gym_fant
from mbrl.envs.benchmarking_envs.gym import gym_cheetahA01
from mbrl.envs.benchmarking_envs.gym import gym_cheetahA003
from mbrl.envs.benchmarking_envs.gym import gym_cheetahO01
from mbrl.envs.benchmarking_envs.gym import gym_cheetahO001
from mbrl.envs.benchmarking_envs.gym import gym_pendulumO01
from mbrl.envs.benchmarking_envs.gym import gym_pendulumO001
from mbrl.envs.benchmarking_envs.gym import gym_cartpoleO01
from mbrl.envs.benchmarking_envs.gym import gym_cartpoleO001
from mbrl.envs.benchmarking_envs.gym import gym_humanoid
from mbrl.envs.benchmarking_envs.gym import gym_nostopslimhumanoid
from mbrl.envs.benchmarking_envs.gym import gym_slimhumanoid


def make_benchmarking_env(id: str):
    envs = {
        'HalfCheetah': HalfCheetahEnv,
        'Walker2D': Walker2dEnv,
        'Ant': AntEnv,
        'Hopper': HopperEnv,
        'Swimmer': SwimmerEnv,
        'FixedSwimmer': gym_fswimmer.fixedSwimmerEnv,
        'FixedWalker': gym_fwalker2d.Walker2dEnv,
        'FixedHopper': gym_fhopper.HopperEnv,
        'FixedAnt': gym_fant.AntEnv,
        'Reacher': ReacherEnv,
        'Pendulum': PendulumEnv,
        'InvertedPendulum': InvertedPendulumEnv,
        'Acrobot': AcrobotEnv,
        'CartPole': CartPoleEnv,
        'MountainCar': Continuous_MountainCarEnv,

        'HalfCheetahO01': gym_cheetahO01.HalfCheetahEnv,
        'HalfCheetahO001': gym_cheetahO001.HalfCheetahEnv,
        'HalfCheetahA01': gym_cheetahA01.HalfCheetahEnv,
        'HalfCheetahA003': gym_cheetahA003.HalfCheetahEnv,

        'PendulumO01': gym_pendulumO01.PendulumEnv,
        'PendulumO001': gym_pendulumO001.PendulumEnv,

        'CartPoleO01': gym_cartpoleO01.CartPoleEnv,
        'CartPoleO001': gym_cartpoleO001.CartPoleEnv,

        'gym_humanoid': gym_humanoid.HumanoidEnv,
        'gym_slimhumanoid': gym_slimhumanoid.HumanoidEnv,
        'gym_nostopslimhumanoid': gym_nostopslimhumanoid.HumanoidEnv,
    }
    env = envs[id]()
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env