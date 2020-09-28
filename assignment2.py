################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
import numpy as np
import random
import pickle
from demo_controller import player_controller
import statistics
import math
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon

from deap import base
from deap import creator
from deap import tools
from deap import algorithms


# Create new directory, if not present yet
experiment_name = 'assignment2c'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Set variables for running
n_hidden_neurons = 10
generations = 100
CXPB = 0.5 # The probability with which two individuals are crossed
MUTPB = 0.5 # The probability for mutating an individual
goal_fit = 101 # Termination fitness level to be reached
pop_size = 100 # Population size for initialization
run_mode = 'train' # Either train, test or plot
save_results = True

# List of enemies to beat. Multiple enemies will result in an iteration of separate training sessions (specialist training)
# Works in both train and test stages, given for testing a pickle file is present for each enemy in the structure: winner_x.pkl where x = enemy number.
enemies = [2, 3, 4] 
n_vars = 21 * n_hidden_neurons + (n_hidden_neurons + 1) * 5

global global_enemy # Define in global scope so we can access it everywhere. Is initialized in the main loop.

def eval_individuals(individual):
    # Main function for evaluating the fitness of individuals in the population
    individual = np.asarray(individual) # Convert to np array so we can reshape if needed
    f, p, e, t = env.play(individual)

    print("Fitness for this individual: {}".format(f))
    return f, # Return as a tuple with second position open


# Initialize the DEAP environment
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize one fitness function
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_individuals)
toolbox.register("mate", tools.cxTwoPoint)

# If mutUniformInt: mutate an individual by replacing attributes, with probability indpb, by a integer uniformly drawn between low and up inclusively. 
# Other possibilities include: mutGaussian, mutPolynomialBounded, mutESLogNormal. See: https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.mutUniformInt
toolbox.register("mutate", tools.mutUniformInt, low = -10, up = 10, indpb = 0.05) # Independent probability of each attribute to be mutated

# There are different selection techniques such as tournament, roulette, selNSGA2, random, best, worst, et cetera.
# Docs: https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament
toolbox.register("select", tools.selTournament, tournsize=3) # Could of course use another selection method


def find_best_individual(pop):
    # Returns fitness and the individual itself for the best individual from the population. 
    # Best here means with the highest fitness value.
    fits = [ind.fitness.values[0] for ind in pop]

    best_fitness = 0
    best_individual = None
    for f, ind in zip(fits, pop):
        if f > best_fitness:
            best_fitness = f
            best_individual = ind

    return best_fitness, best_individual


def main():
    # Define variables for this enemy's loop
    very_best_fitness = 0 # Keeps track of the entire experiment its best fitness value
    very_best_individual = None # Keeps track of the entire experiement its best individual (with best fitness value)
    global env # Make the environment global so we can use the environment in the evaluate function. The environment should be initialized here
    # since in this scope we know the current enemy. The evaluate function cannot be changed w.r.t. its parameters so we define the environment here.

    env = Environment(experiment_name=experiment_name,
              playermode="ai",
              speed="fastest",
              multiplemode="yes",
              enemies = enemies,
              player_controller=player_controller(n_hidden_neurons),
              enemymode="static",
              level=2)

    # Initialize population with shape (n_size, pop_size)
    pop = toolbox.population(n=pop_size)

    fitnesses = list(map(toolbox.evaluate, pop)) # List of tuples: (fitness, )
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit # Set each individual's fitness level for the framework

    best_f, best_i = find_best_individual(pop)
    print("best fitness value is currently: {}".format(best_f))

    fits = [ind.fitness.values[0] for ind in pop]
    g = 0 # Generation counter

    # As long as we haven't reached our fitness goal or the max generations, keep running
    while max(fits) < goal_fit and g < generations:
        temp_best_fitness, temp_best_ind = find_best_individual(pop) # Store best fitness here at the beginning of this generation so we can track improvement for this generation

        g += 1
        print("\n\nGeneration {} of {}, current best fitness is: {}. \n\n".format(g, generations, temp_best_fitness))

        offspring = map(toolbox.clone, toolbox.select(pop, len(pop))) # Select and clone
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB) # Apply crossover and mutation on the offspring

        # Evaluate the individuals with an invalid fitness since for some we deleted their fitness values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring # Replace the pop variable with the new offspring

        best_fitness, best_ind = find_best_individual(pop) 
        if best_fitness > very_best_fitness:
            very_best_fitness = best_fitness
            very_best_individual = best_ind
        print("End of generation, best fitness is now: {}".format(best_fitness))


    # End of the entire run, save best individual
    path = experiment_name + "/best_individual.pkl"
    print("Saving the best individual with fitness: {}".format(very_best_fitness))
    with open(path, "wb") as f:
        pickle.dump(very_best_individual, f)
        f.close()


if __name__ == "__main__":
    if run_mode == 'train':
        main()
    elif run_mode == 'test':
        main_test()