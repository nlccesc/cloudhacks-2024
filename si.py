import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], len(bounds))
        self.velocity = np.random.uniform(-1, 1, len(bounds))
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class RSO:
    def __init__(self, cost_func, bounds, num_particles=30, max_iter=100, alpha=0.1, lr=0.9):
        self.cost_func = cost_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.lr = lr
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(bounds[0], bounds[1], len(bounds))
        self.global_best_score = float('inf')
        self.q_table = np.zeros((num_particles, len(bounds)))
    
    def optimize(self):
        for iteration in range(self.max_iter):
            for i, particle in enumerate(self.swarm):
                fitness = self.cost_func(particle.position)
                if fitness < particle.best_score:
                    particle.best_position = particle.position.copy()
                    particle.best_score = fitness
                
                if fitness < self.global_best_score:
                    self.global_best_position = particle.position.copy()
                    self.global_best_score = fitness
                
                inertia = 0.7
                cognitive_component = 1.4 * np.random.random(len(self.bounds)) * (particle.best_position - particle.position)
                social_component = 1.4 * np.random.random(len(self.bounds)) * (self.global_best_position - particle.position)
                particle.velocity = inertia * particle.velocity + cognitive_component + social_component
                particle.position += particle.velocity

                # Reinforcement Learning update
                reward = -fitness
                self.q_table[i] = (1 - self.lr) * self.q_table[i] + self.lr * (reward + self.alpha * np.max(self.q_table[i]))
    
    def get_best_solution(self):
        return self.global_best_position, self.global_best_score

# Enhanced cost function to simulate server response time
def enhanced_cost_func(params):
    load = params[0]
    if load < 0:
        load = 0  # Ensure non-negative load
    noise = np.random.normal(0, 0.1)
    response_time = np.log(1 + load) + noise
    if np.isnan(response_time) or np.isinf(response_time):
        response_time = 10  # Assign a high response time for invalid cases
    return response_time