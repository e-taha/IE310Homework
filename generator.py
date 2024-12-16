

# Take 4 args (I, J, C, A) 
# (Number of supply nodes, Number of demand nodes, Maximum cost, Maximum demand/supply)

# Generate a random feasible solution first, then generate a a problem instance with that solution

import random
import sys
import numpy as np
import math
import copy
from pulp import *

def generate_random_solution(I, J, C, A):
    # Generate a random feasible solution
    # I: Number of supply nodes
    # J: Number of demand nodes
    # C: Maximum cost
    # A: Maximum demand/supply

    # Generate random supply and demand values
    supply = [A for i in range(I)]
    demand = [A for j in range(J)]

    # Generate random costs
    costs = [[random.randint(0, C) for j in range(J)] for i in range(I)]

    # Generate a random feasible solution
    solution = [[0 for j in range(J)] for i in range(I)]
    for i in range(I):
        for j in range(J):
            if supply[i] > 0 and demand[j] > 0:
                solution[i][j] = random.randint(0, min(supply[i], demand[j]))
                supply[i] -= solution[i][j]
                demand[j] -= solution[i][j]
    
    # Update supply and demand values
    for i in range(I):
        supply[i] = sum(solution[i])
    for j in range(J):
        demand[j] = sum([solution[i][j] for i in range(I)])

    return supply, demand, costs, solution

def generate_problem_instance(I, J, C, A):
    # Generate a problem instance with the given solution
    # I: Number of supply nodes
    # J: Number of demand nodes
    # C: Maximum cost
    # A: Maximum demand/supply

    supply, demand, costs, solution = generate_random_solution(I, J, C, A)
    print("Solution: ")
    print(solution)
    print("Supply: ", supply)
    print("Demand: ", demand)
    print("Costs: ")
    print(costs)

    return supply, demand, costs

def utilize_solver(supply, demand, costs):
    x = [[None for j in range(len(demand))] for i in range(len(supply))]
    for i in range(len(supply)):
        for j in range(len(demand)):
            x[i][j] = LpVariable("x"+str(i)+"_"+str(j), 0, None, LpContinuous)
    prob = LpProblem("Transportation-Problem", LpMinimize)

    # Add Constraints
    for i in range(len(supply)):
        prob += lpSum([x[i][j] for j in range(len(demand))]) == supply[i]
    for j in range(len(demand)):
        prob += lpSum([x[i][j] for i in range(len(supply))]) == demand[j]

    # Add Objective
    prob += lpSum([x[i][j] * costs[i][j] for i in range(len(supply)) for j in range(len(demand))])

    prob.solve()
    print("Status: ", LpStatus[prob.status])
    print("Objective: ", value(prob.objective))
    print("Solution: ")
    for i in range(len(supply)):
        print([value(x[i][j]) for j in range(len(demand))])

# LP Format:
# C = [c_i] (Cost vector)
# Type: Min or Max
# A = [a_ij] (Constraint matrix)
# b = [b_i] (Constraint vector)
# x = [x_i] (Variable vector)
# const_type = [<=, =, >=] (Constraint type vector)
# x_type = [<=, free, >=] (Variable type vector)

# Normalized LP Format:
# C = [c_i] (Cost vector)
# Type: Min or Max
# A = [a_ij] (Constraint matrix)
# b = [b_i] (Constraint vector)
# x = [x_i] (Variable vector)




def standardize_lp(C, A, b, const_type, x_type, type): # const_types will be <= and x_types will be >=0
    # Standardize the LP problem
    pass



if(len(sys.argv) < 5):
    print("Usage: python3 generator.py I J C A")
I, J, C, A = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
supply, demand, costs = generate_problem_instance(int(I), int(J), int(C), int(A))
utilize_solver(supply, demand, costs)