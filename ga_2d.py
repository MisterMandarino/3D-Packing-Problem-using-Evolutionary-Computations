from plot import *

from random import choices, randint, randrange, random
import numpy as np
from tqdm import tqdm

import os
import sys
import time

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process data")
parser.add_argument('-W', '--WIDTH', type=int, required=True, help='The Width of the container')
parser.add_argument('-H', '--HEIGHT', type=int, required=True, help='The Height of the container')
parser.add_argument('-f', '--FILE', type=str, required=True, help='The csv file of the items')
parser.add_argument('-p', '--PLOT', default=False, type=bool, help='Whether to plot the results or not')
parser.add_argument('-v', '--VERBOSE', default=True, type=bool, help='Whether to show items positions')

parser.add_argument('-pop', '--POP_SIZE', default=100, type=int, help='Number of individuals inside a population')
parser.add_argument('-g', '--GEN_LIMIT', default=1000, type=int, help='The maximum number of generations')
parser.add_argument('-n_mut', '--N_MUTATION', default=1, type=int, help='The number of mutations to perform')
parser.add_argument('-p_mut', '--P_MUTATION', default=0.5, type=float, help='Mutation probability')
parser.add_argument('-e', '--N_ELITES', default=2, type=int, help='Number of elites individuals')

args = parser.parse_args()

## Check user's parameters
if args.WIDTH <= 0 or not args.WIDTH:
    parser.error("--WIDTH is required and must be positive")

if args.HEIGHT <= 0 or not args.HEIGHT:
    parser.error("--HEIGHT is required and must be positive")
    
if not args.FILE:
    parser.error("--FILE must be a csv file path containing the items to store")

if args.POP_SIZE <= 0:
    parser.error("--POP_SIZE must be positive")
    
if args.GEN_LIMIT <= 0:
    parser.error("--GEN_LIMIT must be positive")

if args.N_MUTATION <= 0:
    parser.error("--N_MUTATION must be positive")

if args.P_MUTATION < 0 or args.P_MUTATION > 1:
    parser.error("--P_MUTATION must be between 0 and 1")

if args.N_ELITES < 0 or args.N_ELITES >= args.POP_SIZE:
    parser.error("--N_ELITES must be positive and less than the population size")

## Home Directory
HOME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(HOME)

### GLOBAL VARIABLES ###
FILE = args.FILE
W = args.WIDTH
H = args.HEIGHT
PLOT = args.PLOT
VERBOSE = args.VERBOSE

## GA parameters
pop_size = args.POP_SIZE
gen_limit = args.GEN_LIMIT
n_mutations = args.N_MUTATION
p_mutations = args.P_MUTATION
n_elites = args.N_ELITES

## Reading instances
items = []
with open(os.path.join('data',FILE)) as f:
    for index, line in enumerate(f.readlines()):
        separated_values = line.replace('\n', '').split(',')
        separated_values = [int(value) for value in separated_values]

        if len(separated_values) == 2:
            items.append((separated_values[0], separated_values[1]))

## Integrity Check
for item in items:
    assert (item[0] <= W or item[0] <= H) and (item[1] <= W or item[1] <= H), 'The items are too big to fit into the container'
assert sum([item[0]*item[1] for item in items]) <= (W * H), 'The items cannot fit into the container'
        

## Define Genetic Operators
def generate_genome(items):
    candidate = []

    for obj in items:
        # Each object has x, y, width, height as decision variables
        r = randint(0,1)
        if r == 0: ## No rotation
            x = randint(0, W - obj[0])
            y = randint(0, H - obj[1])
        elif r == 1: ## Rotation
            x = randint(0, W - obj[1])
            y = randint(0, H - obj[0])
        candidate.append((x, y, r))
    return candidate

def generate_population(size: int, items):
    return [generate_genome(items) for _ in range(size)]

## Fitness using Matrix vectorization (Maximization problem)
def fitness(genome, items, add_penalty=False):
    items_cpy = rotate_items(genome, items)

    sol_space = np.zeros((W, H), dtype=np.int32)

    for gene, item in zip(genome, items_cpy):
        sol_space[gene[0]:gene[0]+item[0], gene[1]:gene[1]+item[1]] += 1

    good_pos = np.where(sol_space == 1) ## the more space they occupy the better the fitness
    good_fit = len(sol_space[good_pos])

    if add_penalty:
        bad_pos = np.where(sol_space > 1) ## penalty function for overlapped area
        bad_fit = len(sol_space[bad_pos])
        return (good_fit - bad_fit)
    return good_fit

## Selection for Maximization
def selection_pair(population, n_parents : int = 2): ## Roulette-wheel selection
    return choices(population=population, weights=[fitness(genome, items) for genome in population], k=n_parents)

def single_point_crossover(a, b):
    assert len(a) == len(b), 'Genomes must have the same length'

    length = len(a)
    if length < 2:
        return a, b
    
    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome, items, num: int = 1, probability: float = 0.5):
    for _ in range(num):
        index = randrange(len(genome))
        if random() <= probability:
            r = randint(0,1)
            if r == 0:
                x = randint(0, W - items[index][0])
                y = randint(0, H - items[index][1])
            elif r == 1:
                x = randint(0, W - items[index][1])
                y = randint(0, H - items[index][0])
            genome[index] = (x, y, r)
    return genome

def run_evolution(pop_size=100, generation_limit: int = 100, num_mutation = 1, mutation_probability=0.5, n_elites=2):
    
    fitness_limit = sum([item[0]*item[1] for item in items])

    population = generate_population(pop_size, items=items)
    best_fitnesses = []
    worst_fitnesses = []

    for i in tqdm(range(generation_limit)):
        population = sorted(population, key=lambda genome: fitness(genome, items), reverse=True)

        best_fitnesses.append(fitness(population[0], items))
        worst_fitnesses.append(fitness(population[-1], items))
        
        if fitness(population[0], items) == fitness_limit:
            break
        
        next_generation = population[0:n_elites] ## Elitism

        for j in range(int(len(population) /2) - (n_elites//2)):
            parents = selection_pair(population)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, items, num=num_mutation, probability=mutation_probability)
            offspring_b = mutation(offspring_b, items, num=num_mutation, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]
        
        population = next_generation

    population = sorted(population, key=lambda genome: fitness(genome, items), reverse=True)
    best_fitness = fitness(population[0], items)
    best_fitnesses.append(best_fitness)
    worst_fitnesses.append(fitness(population[-1], items))
    
    return population, i, best_fitness, best_fitnesses, worst_fitnesses, fitness_limit

## Solve the 2D Packing Problem
start_time = time.time()
population, generations, fit, best_fits, worst_fits, fit_lim = run_evolution(pop_size=pop_size,
                                                                            generation_limit=gen_limit, 
                                                                            num_mutation=n_mutations, 
                                                                            mutation_probability=p_mutations, 
                                                                            n_elites=n_elites)

if VERBOSE:
    print("---Execution Time: %s seconds ---" % (time.time() - start_time))
    print('number of generations: ', generations)
    print('items: ', items)
    print('best solution: ', population[0])
    print('maximum fitness = ',fit_lim)
    print('best fitness: ', fit)
if PLOT:
    plot_configuration_with_rotation(population[0], items, W, H)
    plot_fitness(best_fits, worst_fits, generations, fit_lim)