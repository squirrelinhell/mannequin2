
import numpy as np
import gym
import gym.spaces

from mannequin import RunningNormalize, Trajectory

class UnboundedActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        low = self.action_space.low
        diff = self.action_space.high - low
        assert (diff > 0.001).all()
        assert (diff < 1000).all()

        def do_step(action):
            action = np.asarray(action, dtype=np.float32)
            action = action.reshape(low.shape)
            action = np.abs((action - 1.0) % 4.0 - 2.0) * 0.5
            action = diff * action + low
            return self.env._step(action)

        self._step = do_step
        self.action_space = gym.spaces.Box(-np.inf, np.inf, low.shape)

class ArgmaxActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        dims = int(self.env.action_space.n)

        def do_step(action):
            action = np.asarray(action, dtype=np.float32)
            action = action.reshape((dims,))
            return self.env._step(np.argmax(action))

        self._step = do_step
        self.action_space = gym.spaces.Box(0.0, 1.0, (dims,))

class NormalizedObservations(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.env.observation_space, gym.spaces.Box)
        normalize = RunningNormalize(self.env.observation_space.shape)

        def do_step(action):
            obs, reward, done, info = self.env._step(action)
            obs = normalize(obs)
            return obs, reward, done, info

        self._step = do_step
        self.get_mean = normalize.get_mean
        self.get_var = normalize.get_var
        self.get_std = normalize.get_std

class PrintRewards(gym.Wrapper):
    def __init__(self, env, every=1000, print=print):
        super().__init__(env)
        ep_rew = None
        finished = []
        total_steps = 0
        mean_rew = 0.0
        def do_step(action):
            nonlocal ep_rew, finished, total_steps, mean_rew
            obs, reward, done, info = self.env._step(action)
            assert ep_rew is not None
            ep_rew += reward
            total_steps += 1
            self.total_steps = total_steps
            if done:
                finished.append(ep_rew)
                ep_rew = None
            if total_steps % every == 0:
                if len(finished) >= 1:
                    mean_rew = np.mean(finished)
                    finished = []
                print(total_steps, mean_rew)
            return obs, reward, done, info
        def do_reset():
            nonlocal ep_rew
            ep_rew = 0.0
            return self.env._reset()
        self.total_steps = 0
        self._step = do_step
        self._reset = do_reset

def one_step(env, policy):
    obs = env.next_obs if hasattr(env, "next_obs") else None
    obs = env.reset() if obs is None else obs
    act = policy(obs)
    next_obs, rew, done, _ = env.step(act)
    env.next_obs = None if done else next_obs
    return obs, act, float(rew), bool(done)

def episode(env, policy, *, render=False, max_steps=0):
    env.next_obs = env.reset()
    if render:
        env.render()

    hist = []
    done = False

    while not done:
        *exp, done = one_step(env, policy)
        hist.append(exp)
        if render:
            env.render()
        if len(hist) == max_steps:
            break

    return Trajectory(*zip(*hist))
