#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin import RunningNormalize, Adam
from mannequin.basicnet import LReLU
from mannequin.gym import episode

def run():
    from _env import build_env, get_progress, mlp_policy
    env = build_env()

    policy = mlp_policy(env, hid_layers=1, activation=LReLU)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=10)

    while get_progress() < 1.0:
        traj = episode(env, policy.sample)
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r), lr=0.001)
        policy.load_params(opt.get_value())

if __name__ == "__main__":
    run()
