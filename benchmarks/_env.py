
import os
import sys
import gym
import gym.spaces

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
                    video_callable=pop_wanted,
                    force=True
                )

            super().__init__(env)

    c = lambda *a, **b: lambda: TrackedEnv(*a, **b)
    configs = {
        "cartpole": c("CartPole-v1", print_every=1000, max_steps=40000),
        "acrobot": c("Acrobot-v1", max_steps=80000),
        "car": c("MountainCar-v0", max_steps=80000, max_rew=200),
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
    policy = Affine(policy, action_size, init=0.1)
    policy = Distribution(policy)

    return policy
