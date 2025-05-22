from deap import base, creator, tools
import random
import numpy as np
from agents.fuzzy_agent import FuzzyAgent
from env.simple_env import SimpleMultiAgentEnv

env = SimpleMultiAgentEnv()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    agent = FuzzyAgent(individual)
    obs = env.reset()
    total_reward = 0

    for _ in range(10):
        actions = {i: agent.act(o) for i, o in obs.items()}
        obs, rewards, _, _ = env.step(actions)
        total_reward += sum(rewards.values())

    return (total_reward,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
