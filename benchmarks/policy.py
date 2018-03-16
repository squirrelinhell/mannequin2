#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("..")

def stochastic_policy(env, *, hid_layers=2, hid_size=64):
    import gym
    from mannequin.basicnet import Input, Affine, Tanh
    from mannequin.distrib import Discrete, Gauss

    if isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.low.size
        Distribution = lambda p: Gauss(mean=p)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
        Distribution = lambda p: Discrete(logits=p)
    else:
        raise ValueError("Unsupported action space")

    policy = Input(env.observation_space.low.size)
    for _ in range(hid_layers):
        policy = Tanh(Affine(policy, hid_size))
    policy = Affine(policy, action_size, init=0.1)
    policy = Distribution(policy)

    return policy

def run():
    from mannequin import RunningNormalize, Adam
    from mannequin.gym import episode
    from _env import build_env, get_progress

    env = build_env()
    policy = stochastic_policy(env)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=2)

    while get_progress() < 1.0:
        traj = episode(env, policy.sample)
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.logprob.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r), lr=0.001)
        policy.load_params(opt.get_value())

if __name__ == "__main__":
    run()
