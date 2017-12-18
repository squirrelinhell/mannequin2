#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("..")
from mannequin.basicnet import Input, Affine, LReLU
from mannequin.logprob import Discrete
from mannequin.gym import PrintRewards, episode

def run():
    from _algo import policy as optimize ### policy / ppo

    print("# steps reward")
    env = gym.make("CartPole-v1")
    env = PrintRewards(env)

    policy = Input(env.observation_space.low.size)
    policy = LReLU(Affine(policy, 64))
    policy = Affine(policy, env.action_space.n)
    policy = Discrete(logits=policy)

    def trajs():
        while env.total_steps < 40000:
            yield episode(env, policy.sample).discounted(horizon=500)

    optimize(logprob=policy, trajs=trajs())

if __name__ == "__main__":
    run()
