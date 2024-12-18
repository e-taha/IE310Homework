import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def parse_results(file_path):
    """
    Parse the experimental results file.
    
    Returns:
    - problem_sizes: List of problem sizes
    - solver1_times: List of solver 1 solve times
    - solver2_times: List of solver 2 solve times
    """
    problem_sizes = []
    solver1_times = []
    solver2_times = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Regular expressions to extract sizes and solve times
    size_pattern = r"Experiment with Size: (\d+)x(\d+)"
    solver1_time_pattern = r"Solve Time: ([\d.]+) seconds"
    
    # Find all matches
    sizes = re.findall(size_pattern, content)
    solver1_times_matches = re.findall(solver1_time_pattern, content)[::2]  # Take every other match for Solver 1
    solver2_times_matches = re.findall(solver1_time_pattern, content)[1::2]  # Take every other match for Solver 2
    
    # Convert to appropriate types
    problem_sizes = [int(size[0]) for size in sizes]
    solver1_times = [float(time) for time in solver1_times_matches]
    solver2_times = [float(time) for time in solver2_times_matches]
    
    return problem_sizes, solver1_times, solver2_times

def analyze_complexity(problem_sizes, solve_times):
    """
    Perform complexity analysis using polynomial regression.
    
    Returns:
    - Coefficients of the best-fit polynomial
    - R-squared value
    """
    # Convert to numpy arrays
    x = np.array(problem_sizes)
    y = np.array(solve_times)
    
    # Try different polynomial degrees
    best_r2 = 0
    best_coeffs = None
    best_degree = 1
    
    for degree in range(1, 4):  # Try linear, quadratic, and cubic
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        
        # Calculate R-squared
        y_pred = p(x)
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        
        if r2 > best_r2:
            best_r2 = r2
            best_coeffs = coeffs
            best_degree = degree
    
    return best_coeffs, best_r2, best_degree

def plot_solver_complexity(file_path):
    """
    Plot solver complexity and perform analysis.
    """
    # Parse results
    problem_sizes, solver1_times, solver2_times = parse_results(file_path)
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))
    
    # Solver 1 Complexity
    plt.subplot(1, 2, 1)
    plt.scatter(problem_sizes, solver1_times, label='Solver 1 Actual Times', color='blue')
    
    # Analyze Solver 1 complexity
    solver1_coeffs, solver1_r2, solver1_degree = analyze_complexity(problem_sizes, solver1_times)
    solver1_poly = np.poly1d(solver1_coeffs)
    
    # Plot Solver 1 fitted curve
    x_smooth = np.linspace(min(problem_sizes), max(problem_sizes), 200)
    plt.plot(x_smooth, solver1_poly(x_smooth), color='red', 
             label=f'Solver 1 Fit (Degree {solver1_degree}, R²={solver1_r2:.4f})')
    
    plt.title('Solver 1 Time Complexity')
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Solve Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    # Solver 2 Complexity
    plt.subplot(1, 2, 2)
    plt.scatter(problem_sizes, solver2_times, label='Solver 2 Actual Times', color='green')
    
    # Analyze Solver 2 complexity
    solver2_coeffs, solver2_r2, solver2_degree = analyze_complexity(problem_sizes, solver2_times)
    solver2_poly = np.poly1d(solver2_coeffs)
    
    # Plot Solver 2 fitted curve
    plt.plot(x_smooth, solver2_poly(x_smooth), color='red', 
             label=f'Solver 2 Fit (Degree {solver2_degree}, R²={solver2_r2:.4f})')
    
    plt.title('Solver 2 Time Complexity')
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Solve Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('solver_complexity.png')
    plt.close()
    
    # Print detailed analysis
    print("Solver 1 Complexity Analysis:")
    print(f"Best-fit Polynomial (Degree {solver1_degree}): {solver1_poly}")
    print(f"R-squared: {solver1_r2:.4f}")
    print("\nSolver 2 Complexity Analysis:")
    print(f"Best-fit Polynomial (Degree {solver2_degree}): {solver2_poly}")
    print(f"R-squared: {solver2_r2:.4f}")
    
    return {
        'solver1': {
            'coeffs': solver1_coeffs,
            'r2': solver1_r2,
            'degree': solver1_degree
        },
        'solver2': {
            'coeffs': solver2_coeffs,
            'r2': solver2_r2,
            'degree': solver2_degree
        }
    }

def main():
    # Assuming the results are in a file named 'experiment_hour_minute.txt'
    import os
    
    # Find the most recent experiment file
    files = [f for f in os.listdir('.') if f.startswith('experiment_') and f.endswith('.txt')]
    if not files:
        print("No experiment results file found.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Analyzing results from: {latest_file}")
    
    # Perform complexity analysis and plotting
    plot_solver_complexity(latest_file)

if __name__ == "__main__":
    main()