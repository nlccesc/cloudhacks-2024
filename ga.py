# ga.py

from deap import base, creator, tools, algorithms
import numpy as np

def genetic_algorithm_optimization(cost_func, bounds):
    # Define the individual and the fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, bounds[0][0], bounds[0][1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(bounds))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=bounds[0][0], up=bounds[0][1], eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Wrap the cost function to return a tuple
    def wrapped_cost_func(individual):
        return (cost_func(individual),)
    
    toolbox.register("evaluate", wrapped_cost_func)

    # Genetic Algorithm parameters
    population_size = 50
    generations = 100

    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind, best_ind.fitness.values[0]
