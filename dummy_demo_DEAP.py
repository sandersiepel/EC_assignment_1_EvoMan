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

from deap import base
from deap import creator
from deap import tools


# Create new directory, if not present yet
experiment_name = 'dummy_demo_DEAP'
if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  speed="fastest",
				  multiplemode="no",
				  player_controller=player_controller(n_hidden_neurons),
				  enemymode="static",
				  level=2)

n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


def eval_individuals(individual):
	individual = np.asarray(individual) # Convert to np array so we can reshape if needed
	f, p, e, t = env.play(individual)
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
toolbox.register("mutate", tools.mutUniformInt, low = -10, up = 10, indpb = 0.1) # Independent probability of each attribute to be mutated

# There are different selection techniques such as tournament, roulette, selNSGA2, random, best, worst, et cetera.
# Docs: https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament
toolbox.register("select", tools.selTournament, tournsize=3) # Could of course use another selection method


# Set variables for running
generations = 100
CXPB = 0.5 # The probability with which two individuals are crossed
MUTPB = 0.2 # The probability for mutating an individual
goal_fit = 90 # Termination fitness level to be reached
pop_size = 50 # Population size for initialization
run_mode = 'train' # Either train or test

# List of enemies to beat. Multiple enemies will result in an iteration of separate training sessions (specialist training)
# Works in both train and test stages, given for testing a pickle file is present for each enemy in the structure: winner_x.pkl where x = enemy number.
enemies = [1, 2, 3, 4, 5, 6, 7] 


def main():
	for e in enemies:
		# Initialize population with shape (n_size, pop_size)
		pop = toolbox.population(n=pop_size)

		fitnesses = list(map(toolbox.evaluate, pop)) # List of tuples: (fitness, )
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit # Set each individual's fitness level for the framework

		fits = [ind.fitness.values[0] for ind in pop] # Plain list of all fitness values (float)
		
		g = 0 # Generation counter

		# As long as we haven't reached our fitness goal or the max generations, keep running
		while max(fits) < goal_fit and g < generations:
			g += 1
			print("Generation {} of {}. Population size: {}".format(g, generations, len(pop)))

			offspring = toolbox.select(pop, len(pop)) 
			
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]): # Loop over all offspring
				if random.random() < CXPB: # Probability the two are going to crossed
					toolbox.mate(child1, child2)
					del child1.fitness.values # Delete their old entries
					del child2.fitness.values

			for mutant in offspring:
				if random.random() < MUTPB: # Are we going to mutate? Try for each individual
					toolbox.mutate(mutant)
					del mutant.fitness.values

			# Evaluate the individuals with an invalid fitness since for some we deleted their fitness values
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

			print('There are {} individuals with no valid fitness function, lets validate them.'.format(len(invalid_ind)))

			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit

			pop[:] = offspring # Replace the pop variable with the new offspring

			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values[0] for ind in pop]


		# Out of while loop, find the winner and save it
		best_f = 0.
		best_p = None
		for f, p in zip(fits, pop):
			if f > best_f:
				best_f = f
				best_p = p

		# Save best
		path = experiment_name + "/winner_"+str(e)+".pkl"
		with open(path, "wb") as f:
		    pickle.dump(best_p, f)
		    f.close()

		print('End of this enemy, best fitness is: {}. Saving solution in pickle file...\n\n'.format(best_f))


def main_test():
	for e in enemies:
		path = experiment_name + "/winner_"+str(e)+".pkl"
		with open(path, "rb") as f:
			winner = pickle.load(f)

		n_hidden_neurons = 10

		env = Environment(experiment_name=experiment_name,
					  playermode="ai",
					  speed="normal",
					  enemies=[e],
					  multiplemode="no",
					  player_controller=player_controller(n_hidden_neurons),
					  enemymode="static",
					  level=2)

		individual = np.asarray(winner) # Convert to np array so we can reshape if needed
		f, p, e, t = env.play(individual)

		print('End of game, fitness was: {}'.format(f))

	
if __name__ == "__main__":
	if run_mode == 'train':
		main()
	elif run_mode == 'test':
		main_test()
