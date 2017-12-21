#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin import RunningNormalize, Adams
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.logprob import Discrete
from mannequin.gym import episode

from _env import cartpole as build_env

def run():
    env = build_env()

    policy = Input(env.observation_space.low.size)
    policy = LReLU(Affine(policy, 64))
    policy = Affine(policy, env.action_space.n)
    policy = Discrete(logits=policy)

    opt = Adams(
        policy.get_params(),
        lr=0.00005,
        horizon=5,
        epsilon=4e-8
    )

    normalize = RunningNormalize(horizon=10)

    while env.progress < 1.0:
        traj = episode(env, policy.sample)
        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r))
        policy.load_params(opt.get_value())

if __name__ == "__main__":
    run()
