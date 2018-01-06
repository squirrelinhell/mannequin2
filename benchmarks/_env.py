
import os
import sys
import gym
import gym.spaces
import numpy as np

sys.path.append("..")
from mannequin import bar
from mannequin.basicnet import Input, Affine, Tanh
from mannequin.logprob import Discrete, Gauss
from mannequin.gym import PrintRewards, ClippedActions

get_progress = None

def build_env():
    finished_steps = 1
    global_max_steps = 1
    video_wanted = False

    global get_progress
    def get_progress(*, divide=True):
        if divide:
            return finished_steps / global_max_steps
        else:
            return finished_steps, global_max_steps

    if "N_VIDEOS" in os.environ:
        n_videos = int(os.environ["N_VIDEOS"])
    else:
        n_videos = 5

    class TrackedEnv(gym.Wrapper):
        def __init__(self, env, *,
                max_steps=400000,
                max_rew=500):
            nonlocal global_max_steps
            global_max_steps = max_steps

            def do_step(action):
                nonlocal finished_steps, video_wanted
                finished_steps += 1
                video_every = max(1, max_steps // n_videos)
                if finished_steps % video_every == video_every // 2:
                    video_wanted = True
                return self.env.step(action)
            self._step = do_step

            def pop_wanted(*_):
                nonlocal video_wanted
                ret, video_wanted = video_wanted, False
                return ret

            if isinstance(env.action_space, gym.spaces.Box):
                env = ClippedActions(env)

            n_lines = 0
            def print_line(s, r):
                nonlocal n_lines
                if n_lines < 100:
                    n_lines += 1
                    if "LOG_FILE" in os.environ:
                        with open(os.environ["LOG_FILE"], "a") as f:
                            f.write("%d %.2f\n" % (s, r))
                            f.flush()
                    else:
                        print("%8d steps:" % s, bar(r, max_rew),
                            flush=True)

            env = PrintRewards(env, print=print_line,
                every=max_steps // 100)

            if "LOG_DIR" in os.environ:
                env = gym.wrappers.Monitor(
                    env,
                    os.environ["LOG_DIR"],
                    video_callable=pop_wanted,
                    force=True
                )

            super().__init__(env)

    class DragCar(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            def do_step(action):
                self.unwrapped.state = (
                    self.unwrapped.state[0] * 0.99,
                    self.unwrapped.state[1]
                )
                return self.env._step(action)
            def do_reset():
                self.env._reset()
                self.unwrapped.state = (
                    np.random.choice(np.array([-0.4, 0.4]) - np.pi/6),
                    0
                )
                return np.array(self.unwrapped.state)
            self._step = do_step
            self._reset = do_reset

    configs = {
        "cartpole": lambda: TrackedEnv(gym.make("CartPole-v1"), max_steps=40000),
        "acrobot": lambda: TrackedEnv(gym.make("Acrobot-v1"), max_steps=80000),
        "car": lambda: TrackedEnv(gym.make("MountainCar-v0"), max_steps=80000, max_rew=200),
        "dragcar": lambda: TrackedEnv(DragCar(gym.make("MountainCar-v0")), max_steps=80000, max_rew=200),
        "walker": lambda: TrackedEnv(gym.make("BipedalWalker-v2"), max_rew=300),
        "lander": lambda: TrackedEnv(gym.make("LunarLanderContinuous-v2")),
    }

    if "ENV" in os.environ:
        return configs[os.environ["ENV"]]

    return configs["cartpole"]

build_env = build_env()

def mlp_policy(env, *, hid_layers=2, hid_size=64, activation=Tanh):
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
        policy = activation(Affine(policy, hid_size))
    policy = Affine(policy, action_size, init=0.1)
    policy = Distribution(policy)

    return policy
