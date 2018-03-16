#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("..")

def build_policy(env, *, hid_layers=2, hid_size=64):
    from mannequin.basicnet import Input, Affine, Tanh

    policy = Input(env.observation_space.low.size)
    for _ in range(hid_layers):
        policy = Tanh(Affine(policy, hid_size))
    policy = Affine(policy, env.action_space.low.size)

    return policy

def run():
    import gym
    from mannequin import RunningNormalize, Adam
    from mannequin.gym import NormalizedObservations, ArgmaxActions, episode
    from _env import build_env, get_progress

    env = build_env()
    env = NormalizedObservations(env)

    if isinstance(env.action_space, gym.spaces.Discrete):
        env = ArgmaxActions(env)

    policy = build_policy(env)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=10)

    while get_progress() < 1.0:
        diff = np.random.randn(policy.n_params) * 0.1
        policy.load_params(opt.get_value() + diff)
        r = normalize(np.sum(episode(env, policy).r))
        opt.apply_gradient(diff * r, lr=0.01)

if __name__ == '__main__':
    run()
