
import os
import sys
import gym
import gym.spaces

from mannequin import bar
from mannequin.gym import PrintRewards, ClippedActions

class Logger(gym.Wrapper):
    def __init__(self, env, log_dir, *, video_every=10000):
        steps = 0
        video_wanted = False
        def do_step(action):
            nonlocal steps, video_wanted
            steps += 1
            if steps % video_every == video_every // 2:
                video_wanted = True
            return self.env.step(action)
        self._step = do_step
        def pop_wanted(*_):
            nonlocal video_wanted
            ret, video_wanted = video_wanted, False
            return ret
        env = gym.wrappers.Monitor(
            env,
            log_dir,
            video_callable=pop_wanted,
            force=True
        )
        super().__init__(env)

class Progress(gym.Wrapper):
    def __init__(self, env, *, max_steps):
        super().__init__(env)
        steps = 0
        def do_step(action):
            nonlocal steps
            steps += 1
            self.progress = steps / max_steps
            return self.env._step(action)
        self._step = do_step
        self.progress = 0.0

def builder(name, print_every=2000, steps=400000, max_rew=100):
    def build():
        env = gym.make(name)
        if isinstance(env.action_space, gym.spaces.Box):
            env = ClippedActions(env)
        if "LOG_DIR" in os.environ:
            env = Logger(env, os.environ["LOG_DIR"],
                video_every=steps // 5)
        if "LOG_FILE" in os.environ:
            def append_line(*a):
                with open(os.environ["LOG_FILE"], "a") as f:
                    f.write(" ".join(str(v) for v in a) + "\n")
                    f.flush()
            env = PrintRewards(env, every=print_every,
                print=append_line)
        else:
            env = PrintRewards(env, every=print_every,
                print=lambda s, r: print(bar(r, max_rew), flush=True))
        return Progress(env, max_steps=steps)
    return build

cartpole = builder("CartPole-v1", print_every=1000, steps=40000, max_rew=500)
acrobot = builder("Acrobot-v1", print_every=2000, steps=80000, max_rew=500)
walker = builder("BipedalWalker-v2", print_every=2048, max_rew=300)
lander = builder("LunarLanderContinuous-v2", print_every=2048, max_rew=500)

class ModuleWrapper(object):
    def __init__(self, inner):
        def get(name):
            if name[:2] == "__":
                return inner.__getitem__(name)
            if "PRINT_ENV_ONLY" in os.environ:
                print("problem:", name)
                sys.exit(0)
            return globals()[name]
        self.get = get
    def __getattr__(self, name):
        return self.get(name)

sys.modules["_env"] = ModuleWrapper(sys.modules["_env"])
