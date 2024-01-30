from plot import *
from mip import *
from itertools import permutations

import time
import os
import sys

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process data")
parser.add_argument('-W', '--WIDTH', type=int, required=True, help='The Width of the container')
parser.add_argument('-H', '--HEIGHT', type=int, required=True, help='The Height of the container')
parser.add_argument('-D', '--DEPTH', type=int, required=True, help='The Depth of the container')
parser.add_argument('-f', '--FILE', type=str, required=True, help='The csv file of the items')
parser.add_argument('-p', '--PLOT', default=False, type=bool, help='Whether to plot the results or not')
parser.add_argument('-m', '--MODE', default=False, type=bool, help='Interactive plotting')
parser.add_argument('-v', '--VERBOSE', default=False, type=bool, help='Whether to show items positions')

args = parser.parse_args()

# Check if NUM_PLOTS is set and PLOT is not
if args.WIDTH <= 0 or not args.WIDTH:
    parser.error("--WIDTH is required and must be positive")

if args.HEIGHT <= 0 or not args.HEIGHT:
    parser.error("--HEIGHT is required and must be positive")

if args.DEPTH <= 0 or not args.DEPTH:
    parser.error("--DEPTH is required and must be positive")
    
if not args.FILE:
    parser.error("--FILE must be a csv file path containing the items to store")

## Home Directory
HOME = os.path.dirname(os.path.realpath(__file__))
sys.path.append(HOME)

### GLOBAL VARIABLES ###
FILE = args.FILE
W = args.WIDTH
H = args.HEIGHT
D = args.DEPTH
PLOT = args.PLOT
MODE = args.MODE
VERBOSE = args.VERBOSE

model = Model()

w, h, d = list(), list(), list()
m = list()

## Reading instances
numberOfVariables = 0
with open(os.path.join('data',FILE)) as f:
    for index, line in enumerate(f.readlines()):
        separated_values = line.replace('\n', '').split(',')
        separated_values = [float(value) for value in separated_values]

        if len(separated_values) == 3:
            possible_permutations = set(permutations(separated_values, 3))
            m.append(len(possible_permutations))

            w.append([value[0] for value in possible_permutations])
            h.append([value[1] for value in possible_permutations])
            d.append([value[2] for value in possible_permutations])

            numberOfVariables += 1

## Create variables
x, y, z = list(), list(), list()
s = list()

left, right, under, over, behind, infront = dict(), dict(), dict(), dict(), dict(), dict()

for i in range(numberOfVariables):
    x.append(model.add_var(name=f'x_{i+1}', var_type=CONTINUOUS, lb=0))
    y.append(model.add_var(name=f'y_{i+1}', var_type=CONTINUOUS, lb=0))
    z.append(model.add_var(name=f'z_{i+1}', var_type=CONTINUOUS, lb=0))

    s.append(list())
    for k in range(m[i]):
        s[i].append(model.add_var(name=f's_{i+1}_{k+1}', var_type=BINARY))

    left[i] = dict()
    right[i] = dict()
    under[i] = dict()
    over[i] = dict()
    behind[i] = dict()
    infront[i] = dict()

    for j in range(i+1, numberOfVariables):
        left[i][j] = model.add_var(name=f'l_{i+1}_{j+1}', var_type=BINARY)
        right[i][j] = model.add_var(name=f'r_{i+1}_{j+1}', var_type=BINARY)
        under[i][j] = model.add_var(name=f'u_{i+1}_{j+1}', var_type=BINARY)
        over[i][j] = model.add_var(name=f'o_{i+1}_{j+1}', var_type=BINARY)
        behind[i][j] = model.add_var(name=f'b_{i+1}_{j+1}', var_type=BINARY)
        infront[i][j] = model.add_var(name=f'f_{i+1}_{j+1}', var_type=BINARY)

## Create the model
model.objective = maximize(xsum(s[i][k] for i in range(numberOfVariables) for k in range(m[i])))

for i in range(numberOfVariables):
    for j in range(i+1, numberOfVariables):
        model += left[i][j] + right[i][j] + under[i][j] + over[i][j] + behind[i][j] + infront[i][j] >= xsum(s[i][k] for k in range(m[i])) + xsum(s[j][k] for k in range(m[j])) - 1
        model += left[i][j] + right[i][j] + under[i][j] + over[i][j] + behind[i][j] + infront[i][j] <= xsum(s[i][k] for k in range(m[i]))
        model += left[i][j] + right[i][j] + under[i][j] + over[i][j] + behind[i][j] + infront[i][j] <= xsum(s[j][k] for k in range(m[j]))

        model += x[i] - x[j] + W * left[i][j] <= W - xsum(s[i][k] * w[i][k] for k in range(m[i]))
        model += x[j] - x[i] + W * right[i][j] <= W - xsum(s[j][k] * w[j][k] for k in range(m[j]))
        model += y[i] - y[j] + H * under[i][j] <= H - xsum(s[i][k] * h[i][k] for k in range(m[i]))
        model += y[j] - y[i] + H * over[i][j] <= H - xsum(s[j][k] * h[j][k] for k in range(m[j]))
        model += z[i] - z[j] + D * behind[i][j] <= D - xsum(s[i][k] * d[i][k] for k in range(m[i]))
        model += z[j] - z[i] + D * infront[i][j] <= D - xsum(s[j][k] * d[j][k] for k in range(m[j]))
    
    model += xsum(s[i][k] for k in range(m[i])) == 1

for i in range(numberOfVariables):
    model += x[i] <= W - xsum(s[i][k] * w[i][k] for k in range(m[i]))
    model += y[i] <= H - xsum(s[i][k] * h[i][k] for k in range(m[i]))
    model += z[i] <= D - xsum(s[i][k] * d[i][k] for k in range(m[i]))

## Solving
model.max_gap = 0.05
model.verbose = 0
start_time = time.time()
status = model.optimize(max_seconds=600)

positions = []
orientations = []

if status == OptimizationStatus.OPTIMAL:
    print('The Problem is FEASIBLE')
    print("---Execution Time: %s seconds ---" % (time.time() - start_time))
    print(f'solution: {model.objective_value}')
    items_orientation = dict()
    for v in model.vars:
        if v.name[0] == "s" and v.x >= 0.98:
            sep_name = v.name.split("_")
            item_name = f"Item {sep_name[1]}"

            items_orientation[item_name] = list()

            items_orientation[item_name].append(w[int(sep_name[1])-1][int(sep_name[2])-1])
            items_orientation[item_name].append(h[int(sep_name[1])-1][int(sep_name[2])-1])
            items_orientation[item_name].append(d[int(sep_name[1])-1][int(sep_name[2])-1])

    print("Items:")
    position = []
    for index, v in enumerate(model.vars):
        
        if v.name[0] == "x":
            sep_name = v.name.split("_")
            print(f"\tItem {sep_name[1]}:")
            # print(f"\t\tposition: [{round(v.x/unit)}, ", end="")
            print(f"\t\tposition: [{v.x}, ", end="")
            position.append(v.x)
        elif v.name[0] == "y":
            print(f"{v.x}, ", end="")
            position.append(v.x)
        elif v.name[0] == "z":
            print(f"{v.x}],")
            position.append(v.x)
            positions.append(position)
            position = []
            sep_name = v.name.split("_")
            print(f"\t\torientation: {items_orientation[f'Item {sep_name[1]}']}")
            orientations.append(items_orientation[f'Item {sep_name[1]}'])
    if VERBOSE:
        print("boxes_dimensions = ", end="")
        print(orientations)
        print("boxes_positions = ", end="")
        print(positions)
    if PLOT:
        if MODE:
            plot_configuration_3d_interactive(positions, orientations, W, H, D)
        else:
            plot_configuration_3d(positions, orientations, W, H, D, multi_color=True)

elif status == OptimizationStatus.INFEASIBLE:
    print('The Problem is INFEASIBLE')
    print("---Execution Time: %s seconds ---" % (time.time() - start_time))