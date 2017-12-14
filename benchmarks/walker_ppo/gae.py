
class GAE(object):
    def __init__(self, env, *,
            gam=0.99, lam=0.95, normalize=True):
        import numpy as np
        from mannequin import Trajectory, SimplePredictor
        from mannequin.gym import one_step

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
                idx = np.random.randint(len(learn_traj), size=64)
                value_predictor.sgd_step(learn_traj[idx], lr=0.0003)

            if normalize:
                return traj.modified(rewards=normalized)
            else:
                return traj

        self.get_chunk = get_chunk
