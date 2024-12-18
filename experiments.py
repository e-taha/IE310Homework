import os
import time
import numpy as np
from generator import generate_problem_instance, utilize_solver
from revised_simplex import LinearProgramSolver

def run_experiments(input_sizes):
    # Create a timestamped filename
    current_time = time.localtime()
    filename = f"experiment_{current_time.tm_hour}_{current_time.tm_min}.txt"
    
    # Open the file in write mode
    with open(filename, 'w') as f:
        # Write header
        f.write("Transportation Problem Experimental Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Run experiments for each input size
        for size in input_sizes:
            # Generate problem instance
            supply, demand, costs = generate_problem_instance(*size)
            
            # Write input details to file
            f.write(f"Experiment with Size: {size[0]}x{size[1]}\n")
            f.write(f"Supply: {supply}\n")
            f.write(f"Demand: {demand}\n")
            f.write("Costs Matrix Dimensions: " + 
                    f"{len(costs)}x{len(costs[0])}\n\n")
            
            # Solver 1 (Original Solver)
            solve1_start = time.time()
            status1, objective1, solution1 = utilize_solver(supply, demand, costs)
            solve1_end = time.time()
            
            # Prepare inputs for Solver 2 (Revised Simplex)
            c = np.concatenate(costs)
            A = np.array([[0 for i in range(len(supply)*len(demand))] 
                          for j in range(len(supply)+len(demand))])
            b = np.array(supply + demand)
            constraints = ["=" for i in range(len(supply)+len(demand))]
            
            # Populate constraint matrix
            for i in range(len(supply)):
                for j in range(len(demand)):
                    A[i][i*len(demand)+j] = 1
            for j in range(len(demand)):
                for i in range(len(supply)):
                    A[j+len(supply)][i*len(demand)+j] = 1
            
            # Solver 2 (Revised Simplex)
            solve2_start = time.time()
            solver2 = LinearProgramSolver(A, b, c, constraints, objective="min")
            solution2, objective2, status2 = solver2.solve()
            solve2_end = time.time()
            
            # Write results to file
            f.write("Solver 1 Results:\n")
            f.write(f"Status: {status1}\n")
            f.write(f"Objective: {objective1}\n")
            f.write(f"Solve Time: {solve1_end - solve1_start} seconds\n\n")
            
            f.write("Solver 2 Results:\n")
            f.write(f"Status: {status2}\n")
            f.write(f"Objective: {objective2}\n")
            f.write(f"Solve Time: {solve2_end - solve2_start} seconds\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
            # Optional: print to console as well
            print(f"Completed experiment for {size[0]}x{size[1]}")
    
    print(f"Experiments completed. Results saved in {filename}")

def main():
    # Example list of input sizes to experiment with
    MAX_COST = 500
    MAX_SUPPLY_DEMAND = 500
    
    # Experimental sizes
    sizes = [
        (2, 2, MAX_COST, MAX_SUPPLY_DEMAND),      # Tiny problem
        (4, 4, MAX_COST, MAX_SUPPLY_DEMAND),      # Very small problem
        (6, 6, MAX_COST, MAX_SUPPLY_DEMAND),      # Somewhat small problem
        (8, 8, MAX_COST, MAX_SUPPLY_DEMAND),      # Another small problem
        (10, 10, MAX_COST, MAX_SUPPLY_DEMAND),    # Small problem
        (20, 20, MAX_COST, MAX_SUPPLY_DEMAND),    # Medium problem
        (30, 30, MAX_COST, MAX_SUPPLY_DEMAND),    # Larger problem
        (40, 40, MAX_COST, MAX_SUPPLY_DEMAND),    # Even larger
        (50, 50, MAX_COST, MAX_SUPPLY_DEMAND),    # Significant size
        (60, 60, MAX_COST, MAX_SUPPLY_DEMAND),    # Pushing computational limits
        (70, 70, MAX_COST, MAX_SUPPLY_DEMAND),    # Large problem
        (80, 80, MAX_COST, MAX_SUPPLY_DEMAND),    # Very large problem
    ]

    run_experiments(sizes)

if __name__ == "__main__":
    main()