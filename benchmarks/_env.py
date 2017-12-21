
import os
import sys
import gym
import gym.spaces

from mannequin import bar
from mannequin.basicnet import Input, Affine, Tanh, Multiplier
from mannequin.logprob import Discrete, Gauss
from mannequin.gym import PrintRewards, ClippedActions

get_progress = None

def build_env():
    finished_steps = 1
    global_max_steps = 1
    video_wanted = False

    global get_progress
    get_progress = lambda: finished_steps / global_max_steps

    class TrackedEnv(gym.Wrapper):
        def __init__(self, gym_name, *,
                print_every=2000,
                max_steps=400000,
                max_rew=500):
            nonlocal global_max_steps
            global_max_steps = max_steps

            env = gym.make(gym_name)

            def do_step(action):
                nonlocal finished_steps, video_wanted
                finished_steps += 1
                if finished_steps % (max_steps // 5) == max_steps // 10:
                    video_wanted = True
                return self.env.step(action)
            self._step = do_step

            def pop_wanted(*_):
                nonlocal video_wanted
                ret, video_wanted = video_wanted, False
                return ret

            if isinstance(env.action_space, gym.spaces.Box):
                env = ClippedActions(env)

            if "LOG_FILE" in os.environ:
                def append_line(*a):
                    with open(os.environ["LOG_FILE"], "a") as f:
                        f.write(" ".join(str(v) for v in a) + "\n")
                        f.flush()
                env = PrintRewards(env, every=print_every,
                    print=append_line)
            else:
                env = PrintRewards(env, every=print_every,
                    print=lambda s, r: print("%8d steps:" % s,
                        bar(r, max_rew), flush=True))

            if "LOG_DIR" in os.environ:
                env = gym.wrappers.Monitor(
                    env,
                    os.environ["LOG_DIR"],
                    video_callable=pop_wanted
                )

            super().__init__(env)

    c = lambda *a, **b: lambda: TrackedEnv(*a, **b)
    configs = {
        "cartpole": c("CartPole-v1", print_every=1000, max_steps=40000),
        "acrobot": c("Acrobot-v1", max_steps=80000),
        "walker": c("BipedalWalker-v2", print_every=2048, max_rew=300),
        "lander": c("LunarLanderContinuous-v2", print_every=2048),
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
    policy = Affine(policy, action_size)
    policy = Multiplier(policy, 0.1)
    policy = Distribution(policy)

    return policy
