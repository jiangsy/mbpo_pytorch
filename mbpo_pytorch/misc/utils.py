from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch

from mbpo_pytorch.envs.wrapped_envs import make_vec_envs, get_vec_normalize
from mbpo_pytorch.misc import logger
from mbpo_pytorch.thirdparty.summary_writer import FixedSummaryWriter as SummaryWriter


def log_and_write(writer: Optional[SummaryWriter], log_infos: List, global_step: int):
    for idx, (name, value) in enumerate(log_infos):
        logger.logkv('{}.'.format(idx) + name.split('/')[-1], value)
        if writer and name.find('/') > -1:
            writer.add_scalar(name, value, global_step=global_step)
    logger.dumpkvs()


def evaluate(actor, env_name, seed, num_episode, eval_log_dir,
             device, norm_reward=False, norm_obs=True, obs_rms=None):
    eval_envs = make_vec_envs(env_name, seed, 1, None, eval_log_dir, device, allow_early_resets=True,
                              norm_obs=norm_obs, norm_reward=norm_reward)
    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None and norm_obs:
        assert obs_rms is not None
        vec_norm.training = False
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []
    eval_episode_lengths = []

    states = eval_envs.reset()
    while len(eval_episode_rewards) < num_episode:
        with torch.no_grad():
            actions = actor.act(states, deterministic=True)['actions']

        states, _, _, infos = eval_envs.step(actions)

        eval_episode_rewards.extend([info['episode']['r'] for info in infos if 'episode' in info])
        eval_episode_lengths.extend([info['episode']['l'] for info in infos if 'episode' in info])

    eval_envs.close()

    return eval_episode_rewards, eval_episode_lengths


# noinspection PyUnresolvedReferences
def set_seed(seed: int, strict=False):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(2 ** 30))
    random.seed(np.random.randint(2 ** 30))
    try:
        torch.cuda.manual_seed_all(np.random.randint(2 ** 30))
        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def get_seed():
    return random.randint(0, 2 ** 32 - 1)


def commit_and_save(proj_dir: str, save_dir: Optional[str] = None, auto_save: bool = False):
    import shutil
    if save_dir and auto_save:
        shutil.copytree(proj_dir, save_dir + '/code', ignore=shutil.ignore_patterns('result', 'data', 'ref'))


def merge_dicts(dicts, merge_fn):
    new_dict = {k: [dic[k] for dic in dicts] for k in dicts[0]}
    new_dict = {k: merge_fn(v) for k, v in new_dict.items()}
    return new_dict


def init_logging(config, hparam_dict):
    import datetime
    current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log')
    eval_log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log_eval')
    save_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'save')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparam_dict, metric_dict={})

    logger.configure(log_dir, None, config.log_email, config.proj_name)
    logger.info('Hyperparms:')
    for key, value in hparam_dict.items():
        logger.log('{:35s}: {}'.format(key, value))

    return writer, log_dir, eval_log_dir, save_dir
