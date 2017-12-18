
import numpy as np

from mannequin import Trajectory, SimplePredictor
from mannequin.gym import one_step

def gae(env, policy, *, steps=100000, gam=0.99, lam=0.95):
    rng = np.random.RandomState()

    # Assuming a continuous observation space
    value_predictor = SimplePredictor(
        env.observation_space.low.size
    )

    chunk = 2048
    hist = []

    for _ in range(steps // chunk):
        while len(hist) < chunk + 1:
            hist.append(one_step(env, policy))

        # Estimate value function for each state
        value = value_predictor.predict(
            [hist[i][0] for i in range(chunk + 1)]
        )

        # Compute advantages
        adv = np.zeros(chunk + 1, dtype=np.float32)
        for i in range(chunk-1, -1, -1):
            adv[i] = hist[i][2] - value[i]
            if not hist[i][3]:
                # The next step is a continuation of this episode
                adv[i] += gam * (value[i+1] + lam * adv[i+1])

        # Build a trajectory with advantages as rewards
        traj = Trajectory(
            [hist[i][0] for i in range(chunk)],
            [hist[i][1] for i in range(chunk)],
            adv[:chunk]
        )
        hist = hist[chunk:]

        # Train the value predictor
        learn_traj = Trajectory(traj.o, (adv + value)[:chunk])
        for _ in range(320):
            idx = rng.randint(len(learn_traj), size=64)
            value_predictor.sgd_step(learn_traj[idx], lr=0.0006)

        yield traj
