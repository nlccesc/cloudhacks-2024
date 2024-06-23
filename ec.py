import numpy as np
from scipy.optimize import differential_evolution

def differential_evolution_optimization(cost_func, bounds):
    result = differential_evolution(cost_func, bounds, strategy='best1bin', maxiter=1000, popsize=25,
                                    tol=1e-6, mutation=(0.5, 1), recombination=0.7, 
                                    seed=42, polish=True, disp=True)
    return result.x, result.fun

if __name__ == "__main__":
    # Simulate server response time
    def cost_func(params):
        load = params[0]
        noise = np.random.normal(0, 0.1)  # Add some noise
        response_time = np.log(1 + load) + noise  # Simulate non-linear increase in response time
        return response_time

    bounds = [(0, 10)] * 2  # Example bounds
    optimized_params, optimized_score = differential_evolution_optimization(cost_func, bounds)
    print('Differential Evolution Optimized parameters:', optimized_params)
    print('Optimized score:', optimized_score)
