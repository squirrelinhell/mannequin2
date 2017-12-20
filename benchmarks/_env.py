
import gym
import gym.spaces
from mannequin.gym import PrintRewards, ClippedActions

def builder(name, print_every=2000, steps=400000):
    def build():
        env = gym.make(name)
        if isinstance(env.action_space, gym.spaces.Box):
            env = ClippedActions(env)
        env = PrintRewards(env, every=print_every)
        return env, lambda: env.total_steps / steps
    return build

cartpole = builder("CartPole-v1", print_every=1000, steps=40000)
walker = builder("BipedalWalker-v2", print_every=2048)
lander = builder("LunarLanderContinuous-v2", print_every=2048)
