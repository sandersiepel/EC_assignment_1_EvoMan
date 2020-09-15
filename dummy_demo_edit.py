################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
import numpy as np
import neat
import random
import pickle


# CONFIGURATION
experiment_name = 'dummy_NEAT'
mode = 'test' # Either train or test. In case of test there should be a pickle file present
generations = 100

# List of enemies to beat. Multiple enemies will result in an iteration of separate training sessions (specialist training)
# Works in both train and test stages, given for testing a pickle file is present for each enemy in the structure: winner_x.pkl where x = enemy number.
enemies = [1, 2, 3, 4, 5, 6, 7] 


if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


class player_controller(Controller):
	def __init__(self, net):
		self.net = net

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs))) # Array with 20 inputs between 0 and 1

		output = self.net.activate(inputs) # Activate the FF NN with the normalized 20 sensory inputs

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]


def eval_genomes(genomes, config):
	# A genome is a neural net, aka a "player". Each genome consists of genes (weights) and nodes.
	for genome_id, genome in genomes:
		genome.fitness = 0.0 # Set a default fitness level
		net = neat.nn.FeedForwardNetwork.create(genome, config) # Create the FF NN, neat.nn.feed_forward.FeedForwardNetwork object

		env = Environment(experiment_name=experiment_name,
						  playermode="ai",
						  enemies=[current_e],
						  multiplemode="no",
					  	  speed="fastest",
					  	  player_controller=player_controller(net),
						  enemymode="static",
						  level=2) # Using level 2 is obligated, do NOT change

		f, p, e, t = env.play()
		
		genome.fitness += f # Update the fitness of this genome for evaluation of the genomes and the population


def run(config_path):
	# Function for running the game in train mode

	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	for e in enemies:

		# Set a global variable of the current enemy so it's accessible in the eval_genomes function. The structure of eval_genomes cannot be changed 
		# so this is a workaround.
		global current_e
		current_e = e

		p = neat.Population(config) # In this case: 20 input nodes, 5 output nodes. 

		p.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		p.add_reporter(stats)

		winner = p.run(eval_genomes, generations) # Fitness function, amount of generations

		# Save the winner in a pickle file
		path = experiment_name + "/winner_"+str(current_e)+".pkl"
		with open(path, "wb") as f:
		    pickle.dump(winner, f)
		    f.close()

		print("Winner is: {}".format(winner))


def run_from_solution(config_path, genome_path, enemy):
	# Function for running the game in test mode (with a given solution)

	# Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Load the one genome that is loaded
    net = neat.nn.FeedForwardNetwork.create(genomes[0][1], config) 

    # Call game with only the loaded genome
    env = Environment(experiment_name=experiment_name,
						  playermode="ai",
						  enemies=[enemy],
						  multiplemode="no",
					  	  speed="normal",
					  	  player_controller=player_controller(net),
						  enemymode="static",
						  level=2)

    f, p, e, t = env.play()

    print('End of game, fitness was: {}'.format(f))


if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")

	if mode == 'train':
		run(config_path)
	elif mode == 'test':
		for e in enemies:
			path = experiment_name + "/winner_"+str(e)+".pkl"
			run_from_solution(config_path, genome_path = path, enemy = e) 

