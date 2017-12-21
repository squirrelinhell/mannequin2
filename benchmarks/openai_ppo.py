#!/usr/bin/env python3

import sys
import gym
import numpy as np

from baselines.common import set_global_seeds, tf_util as U
from baselines import logger
from baselines.ppo1 import mlp_policy, pposgd_simple

def run():
    from _env import build_env, get_progress
    env = build_env()

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(np.random.randint(2**32))
    logger.configure()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(
            name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2
        )

    pposgd_simple.learn(
        env, policy_fn,
        max_timesteps=get_progress(divide=False)[1],
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95, schedule='linear', ### linear / constant
    )

if __name__ == '__main__':
    run()
