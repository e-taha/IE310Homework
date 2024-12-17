from generator import generate_problem_instance, utilize_solver
from revised_simplex import LinearProgramSolver
import time
import numpy as np

def main():
    supply, demand, costs = generate_problem_instance(20,20, 500, 500)
    print("Problem: \n\n")
    print("Supply: ", supply)
    print("Demand: ", demand)
    print("Costs: ", costs)
    print("\n")
    solve1_start = time.time()
    status1, objective1, solution1 = utilize_solver(supply, demand, costs)
    solve1_end = time.time()
    # print("Solver 1: ", status1, objective1, solution1)

    # concatenate all sub arrays of costs into one array
    c = np.concatenate(costs)
    A = np.array([[0 for i in range(len(supply)*len(demand))] for j in range(len(supply)+len(demand))])
    b= np.array(supply + demand)
    constraints = ["=" for i in range(len(supply)+len(demand))]
    for i in range(len(supply)):
        for j in range(len(demand)):
            A[i][i*len(demand)+j] = 1
    for j in range(len(demand)):
        for i in range(len(supply)):
            A[j+len(supply)][i*len(demand)+j] = 1
    solve2_start = time.time()
    solver2 = LinearProgramSolver(A,b,c, constraints, objective="min")
    solution2, objective2, status2 = solver2.solve()
    solve2_end = time.time()

    print("Solver 1: ")
    print("Status: ", status1)
    print("Objective: ", objective1)
    print("Solve Time: ", solve1_end - solve1_start)
    print("Solution: ")
    print(solution1)
    print("\n")
    print("Solver 2: ")
    print("Status: ", status2)
    print("Objective: ", objective2)
    print("Solve Time: ", solve2_end - solve2_start)
    print("Solution: ")
    print(solution2)

    # print("Solver 2: ", status2, objective2, solution2)


if __name__ == "__main__":
    main()