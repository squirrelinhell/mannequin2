#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("..")

def mlp_policy(env, *, hid_layers=2, hid_size=64):
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

def gae(env, policy, *, gam=0.99, lam=0.95):
    from mannequin import Trajectory, SimplePredictor
    from mannequin.gym import one_step

    rng = np.random.RandomState()
    hist = []

    # Assuming a continuous observation space
    value_predictor = SimplePredictor(
        env.observation_space.low.size
    )

    def get_chunk():
        nonlocal hist
        length = 2048

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
            if hist[i][3]:
                # Last step of the episode
                adv[i] = hist[i][2] - value[i]
            else:
                # The next step is a continuation
                adv[i] = hist[i][2] + gam * value[i+1] - value[i]
                adv[i] += gam * lam * adv[i+1]

        # Return a joined trajectory with advantages as rewards
        traj = Trajectory(
            [hist[i][0] for i in range(length)],
            [hist[i][1] for i in range(length)],
            adv[:length]
        )
        hist = hist[length:]

        # Train the value predictor before returning
        learn_traj = Trajectory(traj.o, (adv + value)[:length])
        for _ in range(300):
            idx = rng.randint(len(learn_traj), size=64)
            value_predictor.sgd_step(learn_traj[idx], lr=0.003)

        return traj

    return get_chunk

def run():
    from mannequin import RunningNormalize, Adam
    from mannequin.gym import NormalizedObservations
    from _env import build_env, get_progress

    env = build_env()
    env = NormalizedObservations(env)

    policy = mlp_policy(env)
    opt = Adam(policy.get_params(), horizon=10)

    normalize = RunningNormalize(horizon=2)
    get_chunk = gae(env, policy.sample)

    while get_progress() < 1.0:
        traj = get_chunk()
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        baseline = policy.logprob(traj.o, sample=traj.a)

        for _ in range(300):
            idx = np.random.randint(len(traj), size=64)
            logp, backprop = policy.logprob.evaluate(traj.o[idx],
                sample=traj.a[idx])

            grad = np.exp(logp - baseline[idx]).reshape(-1)
            grad[np.logical_and(grad > 1.2, traj.r[idx] > 0.0)] = 0.0
            grad[np.logical_and(grad < 0.8, traj.r[idx] < 0.0)] = 0.0
            grad *= traj.r[idx]

            opt.apply_gradient(backprop(grad), lr=0.0003)
            policy.load_params(opt.get_value())

if __name__ == '__main__':
    run()
