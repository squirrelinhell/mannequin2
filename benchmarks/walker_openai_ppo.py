#!/usr/bin/env python3

import sys
import gym
import numpy as np

sys.path.append("..")
from mannequin import Adam, Trajectory, SimplePredictor
from mannequin.basicnet import Input, Affine, Tanh
from mannequin.logprob import Gauss
from mannequin.gym import PrintRewards, NormalizedObservations, one_step

class GAE(object):
    def __init__(self, env, *, gam=0.99, lam=0.95):
        rng = np.random.RandomState()
        hist = []

        # Assuming a continuous observation space
        value_predictor = SimplePredictor(
            env.observation_space.low.size
        )

        def normalized(v):
            return (v - np.mean(v)) / max(1e-6, np.std(v))

        def get_chunk(policy, length):
            nonlocal hist
            length = int(length)

            # Run steps in the environment
            while len(hist) < length + 1:
                hist.append(one_step(env, policy))

            # Estimate value function for each state
            value = value_predictor.predict(
                [hist[i][0] for i in range(length + 1)]
            )

            # Compute advantages
            adv = np.zeros(length + 1, dtype=np.float32)
            for i in range(length-1, -1, -1):
                adv[i] = hist[i][2] - value[i]
                if not hist[i][3]:
                    # The next step is a continuation of this episode
                    adv[i] += gam * (value[i+1] + lam * adv[i+1])

            # Return a joined trajectory with advantages as rewards
            traj = Trajectory(
                [hist[i][0] for i in range(length)],
                [hist[i][1] for i in range(length)],
                adv[:length]
            )
            hist = hist[length:]

            # Train the value predictor before returning
            learn_traj = Trajectory(traj.o, (adv + value)[:length])
            for _ in range(320):
                idx = rng.randint(len(learn_traj), size=64)
                value_predictor.sgd_step(learn_traj[idx], lr=0.001)

            return traj.modified(rewards=normalized)

        self.get_chunk = get_chunk

def ppo(env, logprob, *, steps):
    gae = GAE(env)
    opt = Adam(logprob.get_params())

    for i in range(steps // 2048):
        traj = gae.get_chunk(logprob.sample, 2048)
        baseline = logprob(traj.o, sample=traj.a)

        for _ in range(320):
            idx = np.random.randint(len(traj), size=64)
            logp, backprop = logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(logp - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=0.0003)
            logprob.load_params(opt.get_value())

def normed_columns(std):
    def init(inps, outs):
        m = np.random.randn(inps, outs)
        return m * (std / np.sqrt(np.sum(np.square(m), axis=0)))
    return init

def run():
    print("# steps reward")
    env = gym.make("BipedalWalker-v2")
    env = PrintRewards(env, every=2048)
    env = NormalizedObservations(env)

    policy = Input(env.observation_space.low.size)
    policy = Tanh(Affine(policy, 64,
        init=normed_columns(1.0), multiplier=1.0))
    policy = Tanh(Affine(policy, 64,
        init=normed_columns(1.0), multiplier=1.0))
    policy = Affine(policy, env.action_space.low.size,
        init=normed_columns(0.01), multiplier=1.0)
    policy = Gauss(mean=policy)

    ppo(env, policy, steps=400000)

if __name__ == '__main__':
    run()
