#!/usr/bin/env python3

import sys
import gym
import numpy as np

sys.path.append("..")
from mannequin import RunningNormalize, Adam
from mannequin.basicnet import Input, Affine, Tanh
from mannequin.logprob import Gauss
from mannequin.gym import NormalizedObservations, ArgmaxActions, episode

def run():
    from _env import build_env, get_progress, mlp_policy
    env = build_env()
    env = NormalizedObservations(env)

    if isinstance(env.action_space, gym.spaces.Discrete):
        env = ArgmaxActions(env)

    policy = Input(env.observation_space.low.size)
    policy = Tanh(Affine(policy, 64))
    policy = Tanh(Affine(policy, 64))
    policy = Affine(policy, env.action_space.low.size)

    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=10)

    while get_progress() < 1.0:
        diff = np.random.randn(policy.n_params) * 0.1
        policy.load_params(opt.get_value() + diff)
        r = normalize(np.sum(episode(env, policy).r))
        opt.apply_gradient(diff * r, lr=0.01)

if __name__ == '__main__':
    run()
