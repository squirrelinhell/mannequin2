
import numpy as np

import mannequin

class Trajectory(object):
    def __init__(self, observations, actions, rewards=None):
        # Standarize observations
        if isinstance(observations[0], (int, np.integer)):
            observations = np.asarray(observations, dtype=np.int32)
        else:
            observations = np.asarray(observations, dtype=np.float32)
        length = len(observations)
        if length < 1:
            raise ValueError("Cannot create empty trajectory")

        # Standarize actions
        if isinstance(actions[0], (int, np.integer)):
            actions = np.asarray(actions, dtype=np.int32)
        else:
            actions = np.asarray(actions, dtype=np.float32)
        if len(actions) != length:
            raise ValueError("Actions don't match observations")

        # Standarize rewards
        if rewards is None:
            rewards = np.ones(length, dtype=np.float32)
        else:
            rewards = np.asarray(rewards, dtype=np.float32)
            rewards = np.broadcast_to(rewards, (length,))

        # Make all arrays read-only
        observations.setflags(write=False)
        actions.setflags(write=False)
        rewards.setflags(write=False)

        def modified(**kwargs):
            data = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards
            }
            for v in kwargs:
                data[v] = kwargs[v](data[v])
            return Trajectory(**data)

        def discounted(*, horizon):
            return Trajectory(
                observations=observations,
                actions=actions,
                rewards=mannequin.discounted(rewards, horizon=horizon)
            )

        # Public methods
        self.get_length = lambda: length
        self.get_observations = lambda: observations[:]
        self.get_actions = lambda: actions[:]
        self.get_rewards = lambda: rewards[:]
        self.modified = modified
        self.discounted = discounted

    def __getattribute__(self, name):
        if name == "observations"[:len(name)]:
            return self.get_observations()
        if name == "actions"[:len(name)]:
            return self.get_actions()
        if name == "rewards"[:len(name)]:
            return self.get_rewards()
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name[:1] in ("o", "a", "r"):
            raise ValueError("Field '%s' is read-only" % name)
        return super().__setattr__(name, value)

    def __getitem__(self, idx):
        o = self.get_observations()[idx]
        a = self.get_actions()[idx]
        r = self.get_rewards()[idx]
        if len(r.shape) >= 1:
            return Trajectory(o, a, r)
        else:
            return Trajectory([o], [a], [r])

    def __len__(self):
        return self.get_length()

    def __str__(self):
        return "<trajectory: %d x %s -> %s>" % (
            self.get_length(),
            self.get_observations().shape[1:],
            self.get_actions().shape[1:]
        )

    def joined(*ts):
        if len(ts) <= 1:
            ts = ts[0]
        return Trajectory(
            np.concatenate([t.get_observations() for t in ts], axis=0),
            np.concatenate([t.get_actions() for t in ts], axis=0),
            np.concatenate([t.get_rewards() for t in ts], axis=0)
        )
