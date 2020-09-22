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

from deap import base
from deap import creator
from deap import tools


# Create new directory, if not present yet
experiment_name = 'dummy_demo_DEAP'
if not os.path.exists(experiment_name):
	os.makedirs(experiment_name)


# Set variables for running
n_hidden_neurons = 10
generations = 100
CXPB = 0.5 # The probability with which two individuals are crossed
MUTPB = 0.3 # The probability for mutating an individual
goal_fit = 101 # Termination fitness level to be reached
pop_size = 30 # Population size for initialization
run_mode = 'train' # Either train or test
save_results = True
max_stagnation = 10 # Set to very high value if you don't want doomsday
extinction_size = 80 # Size in percentage of the extinction in case of a doomsday. This proportion of the population 
# will be deleted and replaced by newly introduced individuals (randomly initialized just like the original population)

# List of enemies to beat. Multiple enemies will result in an iteration of separate training sessions (specialist training)
# Works in both train and test stages, given for testing a pickle file is present for each enemy in the structure: winner_x.pkl where x = enemy number.
enemies = [1, 2, 3, 4, 5, 6, 7] 
n_vars = 21 * n_hidden_neurons + (n_hidden_neurons + 1) * 5

global global_enemy # Define in global scope so we can access it everywhere. Is initialized in the main loop.

def eval_individuals(individual):
	# Main function for evaluating the fitness of individuals in the population
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


def doomsday(pop):
	# This function takes the population and performs a doomsday (aka extinction) with 80% of the population. This means that
	# 20% is kept (elitism), so if population size = 30, we keep 6 elitism individuals. The other 80% is deleted. For 60% of the deleted individuals, 
	# we create a new one and initialize it randomly. For the other 40% of the deleted individuals we perform a crossover. Candidates for crossover are 
	# those members that survived extinction (20% elitism) together with the newly created random members. 

	# So the calculations are as follows (for population size 30)
	# 80%*0.6 = 48% of random individuals, or 15 individuals
	# 80%*0.4 = 32% of crossover individuals, or 9 individuals
	# 20% of best ones from previous generation, or 6 individuals

	print('Doomsday...')

	# First of all, calculate how many individuals should be kept for elitism. This is 20% of the length of the population. 
	start_amount = math.ceil((len(pop) * 0.2)) # 

	# Now find the start_amount indices with the highest fitness values, so we can keep those
	fits = [ind.fitness.values[0] for ind in pop]
	ind = np.argpartition(fits, kth = -start_amount)[-start_amount:] 

	# With the indices, filter the elitism population and the rest
	pop_elitism = [p for idx, p in enumerate(pop) if idx in ind]
	pop_80percent = [p for idx, p in enumerate(pop) if idx not in ind]

	# Find the elitism their fitness values and print them (optional of course)
	fits_pop = [ind.fitness.values[0] for ind in pop_elitism]
	print("Top 6 elitism fitness values that are kept: {}".format(fits_pop))

	# Calculate the proportions of the 80% individuals that should be deleted. 60% should become new individuals, 40% should crossover
	# First calculate how many individuals should be randomly initialized (new individuals), 60% of the 80%
	size_new_individuals = math.ceil((len(pop_80percent) * 0.6)) 

	# Create the newly initialized population with size: size_new_individuals
	new_pop = toolbox.population(n=size_new_individuals)
	sub_pop = new_pop + pop_elitism # And merge the elitism population with the newly initialized individuals

	# Calculate how many individuals the 40% of 80% is (for crossover)
	size_crossover = len(pop_80percent)-size_new_individuals 

	# Create the new population that should be crossed over with sub_pop
	crossover_pop = toolbox.population(n=size_crossover)

	offspring = toolbox.select(crossover_pop, len(crossover_pop)) 
				
	# Clone the selected individuals
	offspring = list(map(toolbox.clone, offspring))

	# Apply crossover and mutation on the offspring
	for child1, child2 in zip(offspring[::2], offspring[1::2]): # Loop over all offspring
		toolbox.mate(child1, child2)
		del child1.fitness.values # Delete their old entries
		del child2.fitness.values

	# Re-create the pop variable by combining 20% elitism, 48% new individuals and 32% crossover individuals
	pop[:] = sub_pop + offspring

	invalid_ind = [ind for ind in pop if not ind.fitness.valid]
	print("size of population after crossover: {}, of which {} dont have a valid fitness value".format(len(pop), len(invalid_ind)))
	print('Lets validate them')

	# And as usual, validate the individuals that do not have a valid fitness value yet (should be 80% of individuals in pop)
	fitnesses = map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	fits = [ind.fitness.values[0] for ind in pop] # Plain list of all fitness (float)
	print('Result after doomsday, we have {} individuals in the population and their fitnesses are: {}'.format(len(pop), fits))

	return pop


def main():
	
	global_stats = {} # Keeps track of the global stats (avg fitness, max fitness, diversity)
	for i in range(1, 11): # Range should be (1, 11) for 10 runs
		np.random.seed(i)
		stats = {} # To keep track of this round (from the 10 rounds) stats, for all enemies
		for current_enemy in enemies:
			# Define variables for this enemy's loop
			global_enemy = current_enemy
			global env # Make the environment global so we can use the environment in the evaluate function. The environment should be initialized here
			# since in this scope we know the current enemy. The evaluate function cannot be changed w.r.t. its parameters so we define the environment here.

			stagnation_counter = 0
			doomsdays = []
			best_fitness = 0

			env = Environment(experiment_name=experiment_name,
					  playermode="ai",
					  speed="fastest",
					  multiplemode="no",
					  enemies = [global_enemy],
					  player_controller=player_controller(n_hidden_neurons),
					  enemymode="static",
					  level=2)

			# Initialize population with shape (n_size, pop_size)
			pop = toolbox.population(n=pop_size)

			fitnesses = list(map(toolbox.evaluate, pop)) # List of tuples: (fitness, )
			for ind, fit in zip(pop, fitnesses):
				ind.fitness.values = fit # Set each individual's fitness level for the framework

			fits = [ind.fitness.values[0] for ind in pop] # Plain list of all fitness (float)
			best_fitness = max(fits)
			
			g = 0 # Generation counter

			# As long as we haven't reached our fitness goal or the max generations, keep running
			while max(fits) < goal_fit and g < generations:
				if stagnation_counter < max_stagnation:
					g += 1
					print("\n\nRun {}, generation {} of {}. Population size: {}. Doomsdays: {}\n\n".format(i, g, generations, len(pop), doomsdays))

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
					valid_ind = [ind.fitness.values[0] for ind in offspring if ind.fitness.valid]

					# print('There are {} individuals with no valid fitness value (and {} WITH valid fitness value), lets validate them.'.format(len(invalid_ind), len(valid_ind)))
					# print('Fitness values of individuals with a valid fitness value: {}'.format(valid_ind))

					fitnesses = map(toolbox.evaluate, invalid_ind)
					for ind, fit in zip(invalid_ind, fitnesses):
						ind.fitness.values = fit

					pop[:] = offspring # Replace the pop variable with the new offspring

					# Gather all the fitnesses in one list
					fits = [ind.fitness.values[0] for ind in pop]

					if max(fits) > best_fitness: 
						best_fitness = max(fits) # We track best fitness to determine the stagnation process
						stagnation_counter = 0 # Reset stagnation counter since we improved relative to previous population
						print("Resetting stagnation counter to 0 since we found improvement.")
					else:
						# No improvement, track for stagnation process (doomsday)
						stagnation_counter += 1
						print("No improvement, stagnation counter is now: {}".format(stagnation_counter))

					# Calculations
					length = len(pop)
					mean = sum(fits)/length
					sum2 = sum(x*x for x in fits)
					std = abs(sum2 / length - mean**2)**0.5

					# Calculate diversity metric
					# Loop over population and calculate average weight for each weight
					weights = {} # Index:value
					for individual in pop:
					    for idx, w in enumerate(individual): # For each weight
					        if idx in weights:
					            weights[idx] += w
					        else:
					            weights[idx] = w

					avg_weights = {}

					for k, v in weights.items():
					    avg_weights[k] = v/len(pop[0])

					# Again loop over population, for each individual calculate the difference in weight relative to the average weight
					total_error = 0
					for indiv in pop:
						total = 0
						for idx, w in enumerate(indiv):
						    err = abs(indiv[idx] - avg_weights[idx])
						    total += err
					    
						total_error += total # Add to the counter for the entire population

					print("Max fitness: {}, avg fitness: {}, std fitness: {}, total error (diversity): {}".format(max(fits), mean, std, total_error))

					# End of generation, do stats
					data = {"gen": g, "maxFitness": max(fits), "avgFitness": mean, "stdFitness": std, "alltimeMaxFitness": best_fitness, "diversity_error":total_error}
					if current_enemy not in stats:
						stats[current_enemy] = []
						stats[current_enemy].append(data)
					else:
						stats[current_enemy].append(data)

				else:
					# Max stagnation reached, introduce doomsday?
					print("Max stagnation reached, doomsday...")
					doomsdays.append(g)
					pop = doomsday(pop)
					stagnation_counter = 0 # So we can continue the loop


			# Out of while loop, find the winner and save it
			best_f = 0. # Best fitness
			best_p = None # Best individual in population
			for f, p in zip(fits, pop):
				if f > best_f:
					best_f = f
					best_p = p

			if save_results:
				# Save best individual for this enemy, from the latest population
				print("Saving result in pickle file: winner_{}_run{}.pkl".format(current_enemy, i))
				path = experiment_name + "/winner_"+str(current_enemy)+"_run"+str(i)+".pkl"
				with open(path, "wb") as f:
				    pickle.dump(best_p, f)
				    f.close()

			print('End of enemy {}, best fitness is: {}.\n\n'.format(current_enemy, best_f))



		# End of enemy loop, going for next iteration from the 10
		global_stats[i] = stats

	# After all 10 runs are done, plot the stats
	pprint.pprint(global_stats)

	# Save the stats to a text file
	path = experiment_name + "/global_stats.txt"
	with open(path, 'wb') as f:
		pickle.dump(global_stats, f)

	getPlots(global_stats)



def main_test():
	# The main function when run_mode = test
	for en in enemies:
		path = experiment_name + "/winner_"+str(en)+".pkl"
		with open(path, "rb") as f:
			winner = pickle.load(f)

		n_hidden_neurons = 10

		env = Environment(experiment_name=experiment_name,
					  playermode="ai",
					  speed="normal",
					  enemies=[en],
					  multiplemode="no",
					  player_controller=player_controller(n_hidden_neurons),
					  enemymode="static",
					  level=2)

		individual = np.asarray(winner) # Convert to np array so we can reshape if needed
		f, p, e, t = env.play(individual)

		print('End of game, fitness was: {}'.format(f))


def getPlots(global_stats):
    for enemy in global_stats.get(1):
        df = pd.DataFrame()
        for x in global_stats:
            df = df.append(global_stats.get(x).get(enemy), ignore_index = True)
            ind = 1 + (df.index / 10)
            df['run'] = ind.astype(int)
            #df.set_index("run", inplace = True)
        df3 = df.groupby("gen").std()
        
        fileName = "enemy" + str(enemy) +".png"
        df.set_index("gen", inplace = True)
        
        avg = pd.DataFrame()
        std = pd.DataFrame()
        avg[['maximum_mean', 'average_mean']] = df[['maxFitness', 'avgFitness']]
        std[['maximum_std', 'average_std']] = df[['maxFitness', 'avgFitness']]

        plt.figure(figsize=(10,6))
        ax = sns.lineplot(
            data=std, estimator = "std",
            hue="event", dashes = [(2, 2), (2, 2)]
           )
        sns.lineplot(
            data=avg, estimator = "mean",
            hue="event", style="event", dashes=False
           )
        
        ax.set_title("Enemy: " + str(enemy))
        ax.set_ylabel("Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylim(-5,100)

        path = experiment_name + "/enemy" + str(enemy) + ".png"
        ax.get_figure().savefig(path)

	
if __name__ == "__main__":
	if run_mode == 'train':
		main()
	elif run_mode == 'test':
		main_test()
