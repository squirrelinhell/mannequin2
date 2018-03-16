#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append("..")

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
    from policy import stochastic_policy

    env = build_env()
    env = NormalizedObservations(env)

    policy = stochastic_policy(env)
    opt = Adam(policy.get_params(), horizon=10)
    normalize = RunningNormalize(horizon=2)
    get_chunk = gae(env, policy.sample)

    while get_progress() < 1.0:
        traj = get_chunk()
        traj = traj.modified(rewards=normalize)
        traj = traj.modified(rewards=np.tanh)

        _, backprop = policy.logprob.evaluate(traj.o, sample=traj.a)
        opt.apply_gradient(backprop(traj.r), lr=0.01)
        policy.load_params(opt.get_value())

if __name__ == '__main__':
    run()
