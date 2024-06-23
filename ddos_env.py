# ddos_env.py
import gym
from gym import spaces
import numpy as np
from ec import differential_evolution_optimization
from si import RSO, enhanced_cost_func
from ga import genetic_algorithm_optimization

class DDoSMitigationEnv(gym.Env):
    def __init__(self):
        super(DDoSMitigationEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Three strategies: Differential Evolution, Swarm Intelligence, Genetic Algorithm
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.state = np.random.rand(3)

    def reset(self):
        self.state = np.random.rand(3)
        return self.state

    def step(self, action):
        context = self.state
        if action == 0:
            params, score = differential_evolution_optimization(enhanced_cost_func, [(0, 10)] * 2)
        elif action == 1:
            rso = RSO(enhanced_cost_func, bounds=[(0, 10)] * 2, num_particles=50, max_iter=200)
            rso.optimize()
            params, score = rso.get_best_solution()
        elif action == 2:
            params, score = genetic_algorithm_optimization(enhanced_cost_func, [(0, 10)] * 2)
        
        reward = -score
        done = True
        self.state = np.random.rand(3)
        return self.state, reward, done, {}

    def render(self, mode='human', close=False):
        pass
