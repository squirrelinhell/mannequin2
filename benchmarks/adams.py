#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin import RunningNormalize, Adams
from mannequin.basicnet import LReLU
from mannequin.gym import episode

def run():
    from _env import build_env, get_progress, mlp_policy
    env = build_env()

    policy = mlp_policy(env, hid_layers=1, activation=LReLU)

    opt = Adams(
        policy.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    while get_progress() < 1.0:
        traj = episode(env, policy.sample)
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r))
        policy.load_params(opt.get_value())

if __name__ == "__main__":
    run()
