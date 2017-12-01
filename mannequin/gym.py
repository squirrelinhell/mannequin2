
import gym

class UnboundedActions(gym.Wrapper):
    def __init__(self, env):
        import numpy as np
        import gym.spaces

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
        import numpy as np
        import gym.spaces

        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.Discrete)
        dims = int(env.action_space.n)

        def do_step(action):
            action = np.asarray(action, dtype=np.float32)
            action = action.reshape((dims,))
            return self.env._step(np.argmax(action))

        self._step = do_step
        self.action_space = gym.spaces.Box(0.0, 1.0, (dims,))

class PrintRewards(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        print("# episode reward", flush=True)
        rew_sum = None
        ep_number = 0

        def do_step(action):
            nonlocal rew_sum
            obs, reward, done, info = self.env._step(action)
            assert rew_sum is not None
            rew_sum += reward
            if done:
                print("%d %.4f" % (ep_number, rew_sum), flush=True)
                rew_sum = None
            return obs, reward, done, info

        def do_reset():
            nonlocal rew_sum, ep_number
            rew_sum = 0.0
            ep_number += 1
            return self.env._reset()

        self._step = do_step
        self._reset = do_reset

def episode(env, policy, *, render=False):
    import gym.spaces
    from mannequin import Trajectory

    assert isinstance(env.action_space, gym.spaces.Box)

    next_obs = env.reset()
    done = False

    all_obs = []
    all_act = []
    all_rew = []

    if render:
        env.render()

    while not done:
        all_obs.append(next_obs)
        all_act.append(policy(next_obs))
        next_obs, rew, done, _ = env.step(all_act[-1])
        all_rew.append(rew)

        if render:
            env.render()

    return Trajectory(all_obs, all_act, all_rew)
