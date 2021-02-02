import os

import argparse
import munch
import yaml
from yaml import Loader
import collections

from mbpo_pytorch.misc import logger


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def safe_eval(exp: str):
    try:
        return eval(exp)
    except (NameError, SyntaxError):
        return exp


def deflatten_with_eval(d, sep='.'):
    deflattend_d = {}
    for k, v in d.items():
        d = deflattend_d
        key_seq = k.split(sep)
        for key in key_seq[:-1]:
            try:
                d = d[key]
            except (TypeError, KeyError):
                d[key] = {}
                d = d[key]
        d[key_seq[-1]] = safe_eval(v)
    return deflattend_d


class Config:
    def __new__(cls, config_paths='config.yaml'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--configs', nargs='+', default=[])
        parser.add_argument('--set', type=str, nargs='*', action='append')

        args, unknown = parser.parse_known_args()
        flattened_config_dict = {}
        overwritten_config_dict = {}

        if args.configs:
            config_paths = args.configs

        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            if not config_path.startswith('/'):
                config_path = os.path.join(os.path.dirname(__file__), config_path)
            logger.info('Loading configs from {}.'.format(config_path))

            with open(config_path, 'r', encoding='utf-8') as f:
                new_config_dict = yaml.load(f, Loader=Loader)
                flattened_new_config_dict = flatten(new_config_dict)
                overwritten_config_dict.update(
                    {k: v for k, v in flattened_new_config_dict.items()
                     if (k in flattened_config_dict.keys() and v != flattened_config_dict[k])})
                flattened_config_dict.update(flattened_new_config_dict)

        if args.set:
            for instruction in sum(args.set, []):
                key, value = instruction.split('=')
                flattened_config_dict.update({key: safe_eval(value)})
                # values set by args should be recorded all
                overwritten_config_dict.update({key: safe_eval(value)})

        config_dict = deflatten_with_eval(flattened_config_dict)

        for key, value in overwritten_config_dict.items():
            logger.notice('Hyperparams {} has been overwritten to {}.'.format(key, value))

        config = munch.munchify(config_dict)
        config_dict = flatten(config_dict)
        logged_config_dict = {}

        for key, value in config_dict.items():
            if key.find('.') >= 0:
                logged_config_dict[key] = value
        return config, logged_config_dict
