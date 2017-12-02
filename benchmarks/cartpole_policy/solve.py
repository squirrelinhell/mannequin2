#!/usr/bin/env python3

import sys
import numpy as np
import gym

sys.path.append("../..")
from mannequin import Trajectory, RunningNormalize, Adams
from mannequin.gym import ArgmaxActions, PrintRewards, episode
from mannequin.basicnet import Input, Affine, LReLU

def softmax(v):
    v = v.T
    v = np.exp(v - np.amax(v, axis=0))
    v /= np.sum(v, axis=0)
    return v.T

def run():
    env = ArgmaxActions(
        PrintRewards(gym.make("CartPole-v1"))
    )

    model = Input(env.observation_space.low.size)
    model = LReLU(Affine(model, 64))
    model = Affine(model, env.action_space.low.size)

    rng = np.random.RandomState()
    opt = Adams(
        rng.randn(model.n_params) * 0.1,
        lr=0.00004,
        horizon=5
    )

    def stochastic_policy(obs):
        policy = softmax(model.evaluate([obs])[0])[0]
        return np.eye(model.n_outputs)[
            rng.choice(model.n_outputs, p=policy)
        ]

    normalize = RunningNormalize(horizon=10)

    for _ in range(150):
        model.load_params(opt.get_value())
        traj = episode(env, stochastic_policy)

        traj = traj.discounted(horizon=500)
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        model_outs, backprop = model.evaluate(traj.o)
        model_outs = softmax(model_outs)
        grad = ((traj.a - model_outs).T * traj.r.T).T
        opt.apply_gradient(backprop(grad))

        trajs = []

if __name__ == "__main__":
    run()
