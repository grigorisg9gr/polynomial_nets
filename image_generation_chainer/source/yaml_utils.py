# !/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import splitext, basename, dirname, isfile, join
from os import getcwd
import sys
import time
from pathlib import Path
from copy import deepcopy

import yaml


# Copy from tgans repo.
class Config(object):
    def __init__(self, config_dict):
        self.config = config_dict

    def __getattr__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise AttributeError(key)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)


def load_dataset(config, validation=False, valid_path=None):
    args = deepcopy(config.dataset['args'])
    dataset = load_module(config.dataset['dataset_fn'],
                          config.dataset['dataset_name'])
    if validation:
        # # Modify the path for the validation db.
        if 'path' in args.keys():
            p1 = args['path']
            if not isfile(p1):
                # # try one more time by appending the current path.
                p1 = join(getcwd(), args['path'])
            if valid_path is not None:
                args['path'] = Path(p1).with_name(valid_path).as_posix()
        else:
            # # e.g. used in cifar10/svhn.
            args['train'] = False
    return dataset(**args)


def load_module(fn, name):
    mod_name = splitext(basename(fn))[0]
    mod_path = dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_fn, model_name, args=None, GPU=0):
    model = load_module(model_fn, model_name)
    model1 = model(**args) if args else model()
    if GPU:
        model1.to_gpu()
    return model1


def load_updater_class(config):
    return load_module(config.updater['fn'], config.updater['name'])

