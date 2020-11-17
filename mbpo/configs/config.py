from typing import List
import os

import argparse
import munch
import yaml
from yaml import Loader
import collections

from mbrl.misc import logger


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def change_dict_value_recursive(obj, key_sequence: List[str], value):
    try:
        for key in key_sequence[:-1]:
            obj = obj[key]
        obj[key_sequence[-1]] = value
    except KeyError:
        raise KeyError('Incorrect key sequences')


class Config:
    def __new__(cls, config_paths='config.yaml'):
        parser = argparse.ArgumentParser(description='Stochastic Lower Bound Optimization')
        parser.add_argument('-c', '--configs', type=str, help='configuration file (YAML)', nargs='+', action='append')
        parser.add_argument('-s', '--set', type=str, help='additional options', nargs='*', action='append')

        args, unknown = parser.parse_known_args()
        config_dict = {}
        flattened_config_dict = {}
        overwritten_config_dict = {}

        if args.configs:
            config_paths = args.configs

        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            logger.info('Loading configs from {}.'.format(config_path))
            if not config_path.startswith('/'):
                config_path = os.path.join(os.path.dirname(__file__), config_path)

            with open(config_path, 'r', encoding='utf-8') as f:
                new_config_dict = yaml.load(f, Loader=Loader)
                flattened_new_config_dict = flatten(new_config_dict)
                overwritten_config_dict.update({k: v for k, v in flattened_new_config_dict.items()
                                                if k in flattened_config_dict.keys() & flattened_new_config_dict})
                config_dict.update(new_config_dict)
                flattened_config_dict.update(flattened_new_config_dict)

        if args.set:
            for instruction in sum(args.set, []):
                key, value = instruction.split('=')
                change_dict_value_recursive(config_dict, key.split('.'), eval(value))
                overwritten_config_dict.update({key: eval(value)})

        for key, value in overwritten_config_dict.items():
            logger.notice('Hyperparams {} has been overwritten to {}.'.format(key, value))

        config = munch.munchify(config_dict)
        config_dict = flatten(config_dict)
        logged_config_dict = {}

        for key, value in config_dict.items():
            if key.find('.') >= 0:
                logged_config_dict[key] = value
        return config, logged_config_dict


