import numpy as np
import copy


class LinearProgramSolver:
    def __init__(self, c, A, b, constraint_types=None, variable_types=None, objective='max'):
        """
        Initialize Linear Programming Problem
        
        Parameters:
        c: Objective function coefficients
        A: Constraint matrix
        b: Right-hand side of constraints
        constraint_types: List of constraint types ('<=', '>=', '=')
        variable_types: List of variable types ('non-negative', 'unrestricted', 'non-positive')
        objective: 'max' or 'min'
        """
        # Convert inputs to numpy arrays
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        
        # Set default constraint types if not provided
        if constraint_types is None:
            constraint_types = ['<='] * len(b)
        self.constraint_types = constraint_types
        
        # Set default variable types (non-negative)
        if variable_types is None:
            variable_types = ['non-negative'] * len(c)
        self.variable_types = variable_types
        
        # Objective type
        self.objective = objective
        
        # Store original problem dimensions
        self.m, self.n = self.A.shape
    
    def _preprocess_problem(self):
        """
        Preprocess the problem:
        1. Flip constraints with negative right-hand side
        2. Handle non-negative and free variables
        
        Returns:
        Modified A matrix, b vector, c vector, and variable mapping
        """
        # Make a copy of original data to modify
        A = copy.deepcopy(self.A)
        b = copy.deepcopy(self.b)
        c = copy.deepcopy(self.c)
        constraint_types = copy.deepcopy(self.constraint_types)
        
        # Flip constraints with negative RHS
        for i in range(len(b)):
            if b[i] < 0:
                # Flip the constraint and multiply row by -1
                A[i, :] *= -1
                b[i] *= -1
                
                # Also flip the constraint type
                if constraint_types[i] == '<=':
                    constraint_types[i] = '>='
                elif constraint_types[i] == '>=':
                    constraint_types[i] = '<='
        
        # Handle variable transformations
        variable_mapping = []
        new_A = []
        new_c = []
        
        for j, var_type in enumerate(self.variable_types):
            if var_type == 'non-negative':
                # No change needed
                variable_mapping.append((j, None))
                new_A.append(A[:, j])
                new_c.append(c[j])
            
            elif var_type == 'non-positive':
                # Convert x to -x where x is non-negative
                variable_mapping.append((j, 'negate'))
                new_A.append(-A[:, j])
                new_c.append(-c[j])
            
            elif var_type == 'unrestricted':
                # x = x1 - x2, where x1, x2 are non-negative
                x1_col = A[:, j].copy()
                x2_col = -A[:, j].copy()
                
                new_A.append(x1_col)
                new_A.append(x2_col)
                
                new_c.append(c[j])
                new_c.append(-c[j])
                
                variable_mapping.append((j, 'split'))
        
        # Convert back to numpy arrays
        new_A = np.column_stack(new_A)
        new_c = np.array(new_c)
        
        return new_A, b, new_c, constraint_types, variable_mapping
    
    def _convert_to_standard_form(self, A, b, c, constraint_types):
        """
        Convert the problem to standard form
        
        Parameters:
        A: Constraint matrix
        b: Right-hand side vector
        c: Objective function coefficients
        constraint_types: List of constraint types
        
        Returns:
        Modified A matrix, b vector, and c vector
        """
        # Handle objective function
        if self.objective == 'min':
            c = copy.deepcopy(-c)
        else:
            c = copy.deepcopy(c)
        
        # Create a copy of constraint matrix and RHS
        A = copy.deepcopy(A)
        b = copy.deepcopy(b)
        
        # Track added slack/surplus variables
        basic_var_indices = []
        artificial_var_indices = []

        # Adjust constraints to standard form
        for i, constraint_type in enumerate(constraint_types):
            if constraint_type == '<=':
                # Less than or equal: add non-negative slack variable
                slack_col = np.zeros(len(b))
                slack_col[i] = 1
                basic_var_indices.append(A.shape[1])
                A = np.column_stack([A, slack_col])
                c = np.append(c, 0)  # Zero coefficient for slack variable
            
            elif constraint_type == '>=':
                # Greater than or equal: subtract surplus variable
                surplus_col = np.zeros(len(b))
                surplus_col[i] = -1
                A = np.column_stack([A, surplus_col])
                c = np.append(c, 0)  # Zero coefficient for surplus variable
                
                # Add artificial variable
                artificial_col = np.zeros(len(b))
                artificial_col[i] = 1
                basic_var_indices.append(A.shape[1])
                artificial_var_indices.append(A.shape[1])
                A = np.column_stack([A, artificial_col])
                c = np.append(c, 0)  # Zero coefficient for artificial variable
            
            elif constraint_type == '=':
                # Equality: add artificial variable
                artificial_col = np.zeros(len(b))
                artificial_col[i] = 1
                basic_var_indices.append(A.shape[1])
                artificial_var_indices.append(A.shape[1])
                A = np.column_stack([A, artificial_col])
                c = np.append(c, 0)  # Zero coefficient for artificial variable
        
        return A, b, c, basic_var_indices, artificial_var_indices
    
    def _two_phase_simplex(self, A, b, c, basic_var_indices, artificial_cols=None):
        """
        Two-Phase Simplex Method
        
        Parameters:
        A: Constraint matrix in standard form
        b: Right-hand side vector
        c: Objective function coefficients
        
        Returns:
        Optimal solution, objective value, status
        """
        m, n = A.shape
        
        # Phase I: Minimize sum of artificial variables
        phase1_c = np.zeros(n)
        phase1_c[artificial_cols] = 1
        
        # Initial basic variables are artificial variables and slack variables
        initial_basis = basic_var_indices
        
        # Solve Phase I
        phase1_solution, phase1_obj, phase1_status, soln_basis = self._revised_simplex(
            A, b, phase1_c, 
            initial_basis=initial_basis, 
            minimize=True
        )
        
        # Check feasibility
        if phase1_status != 'Optimal' or abs(phase1_obj) > 1e-8 or np.any(np.isin(np.array(soln_basis), np.array(artificial_cols))):
            return None, None, 'Infeasible'
        
        # Phase II: Solve original problem
        solution, obj_value, status, second_soln_basis = self._revised_simplex(
            A, b, c, 
            initial_basis=soln_basis, 
            minimize=(self.objective == 'min')
        )
        
        return solution, obj_value, status
    

    def construct_E(self, B_inv, a_col, r):
        m = B_inv.shape[0]
        E = np.eye(m)  # Start with the identity matrix
        print("a_col: ", a_col)
        print("r: ", r)
        # print("B_inv: ", B_inv)
        # Compute the r-th column of E
        pivot = a_col[r]
        for i in range(m):
            if i == r:
                E[i, r] = 1 / pivot  # Update pivot row
            else:
                E[i, r] = -a_col[i] / pivot  # Update other rows
        return E
    
    def _revised_simplex(self, A, b, c, initial_basis=None, minimize=False, max_iterations=100):
        """
        Revised Simplex Method solver
        
        Parameters:
        A: Constraint matrix
        b: Right-hand side vector
        c: Objective function coefficients
        initial_basis: Initial basic variable indices
        minimize: Objective direction
        
        Returns:
        Solution, objective value, status
        """
        m, n = A.shape
        tolerance = 1e-8
        
        # Initialize basis
        if initial_basis is None:
            basic_vars = list(range(n, n + m))
            non_basic_vars = list(range(n))
            B = np.eye(m)
        else:
            basic_vars = initial_basis
            non_basic_vars = [i for i in range(n) if i not in basic_vars]
            B = A[:, basic_vars]
        
        # Objective coefficient adjustment for minimization
        obj_multiplier = -1 if minimize else 1
        adj_c = obj_multiplier * c
        
        for i in range(max_iterations):
            # Compute basic solution
            if i % 30 == 0:
                try:
                    B_inv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    return None, None, 'Singular Matrix', None
            else:
                E = self.construct_E(B_inv, A[:, entering_idx], leaving_index_x)
                B_inv = E @ B_inv
            x_B = np.dot(B_inv, b)
            # Compute reduced costs
            print("B ")
            print(B)
            print("B_inv ")
            print(B_inv)
            print("A ")
            print(A)
            # print("adj_c ", adj_c)
            print("x_B ")
            print(x_B)
            print("basic_vars ", basic_vars)
            print("non_basic_vars ", non_basic_vars)

            
            reduced_costs = np.dot(np.dot(adj_c[basic_vars], B_inv), A) - adj_c
            print("reduced_costs: ")    
            print(reduced_costs)

            input("Press Enter to continue to the next iteration...")
            # Check for optimality
            if np.all(reduced_costs >= -tolerance):
                # Construct full solution vector
                x = np.zeros(n)
                for i, var in enumerate(basic_vars):
                    if var < n:
                        x[var] = x_B[i]
                
                obj_value = obj_multiplier * np.dot(c, x)
                return x, obj_value, 'Optimal', basic_vars
            
            # Select entering variable (most negative reduced cost)
             
            entering_idx = np.argmin(reduced_costs)
            entering_index_x = np.where(non_basic_vars == entering_idx)[0][0]

            # Compute pivot column
            try:
                pivot_col = np.dot(B_inv, A[:, entering_idx])
            except np.linalg.LinAlgError:
                return None, None, 'Numerical Error', None
            
            # Check for unboundedness
            if np.all(pivot_col <= tolerance):
                return None, None, 'Unbounded', None
            
            # Compute ratios for leaving variable

            ratios = np.where(pivot_col > tolerance, x_B / pivot_col, np.inf)
            leaving_index_x = np.argmin(ratios)
            leaving_idx = basic_vars[leaving_index_x]

            
            # Update basis
            B[:, leaving_index_x] = A[:, entering_idx]

            # Update basic and non-basic variable lists
            print("entering_idx: ", entering_idx)
            print("leaving_idx: ", leaving_idx)
            basic_vars[leaving_index_x] = entering_idx
            non_basic_vars[entering_index_x] = leaving_idx
            print("basic_vars: ", basic_vars)
            print("non_basic_vars: ", non_basic_vars)
        
        return None, None, 'Max Iterations Reached', None
    
    def solve(self):
        """
        Solve the linear programming problem
        
        Returns:
        Optimal solution, objective value, status
        """
        # Preprocess the problem (handle negative RHS and variable types)
        A_preprocessed, b_preprocessed, c_preprocessed, constraint_types, variable_mapping = self._preprocess_problem()
        
        # Convert to standard form
        A_std, b_std, c_std, basic_var_indices, artificial_var_indices = self._convert_to_standard_form(
            A_preprocessed, b_preprocessed, c_preprocessed, constraint_types
        )
        
        # Solve using two-phase simplex method
        solution, obj_value, status = self._two_phase_simplex(A_std, b_std, c_std, basic_var_indices, artificial_var_indices)
        
        # Reconstruct original solution if needed
        if solution is not None:
            original_solution = np.zeros(self.n)
            j_orig = 0
            for j, mapping in enumerate(variable_mapping):
                if mapping[1] is None:
                    # Non-negative variable
                    original_solution[j] = solution[j_orig]
                    j_orig += 1
                elif mapping[1] == 'negate':
                    # Non-positive variable
                    original_solution[j] = -solution[j_orig]
                    j_orig += 1
                elif mapping[1] == 'split':
                    # Unrestricted variable
                    original_solution[j] = solution[j_orig] - solution[j_orig + 1]
                    j_orig += 2
            
            solution = original_solution
        
        return solution, obj_value, status

def main():
    # Example 1: Problem with negative RHS
    print("Example 1: Negative RHS")
    c = np.array([2, 7])
    A = np.array([
        [2, 1],    # x + y ≤ -5
        [4, 2]     # 2x + y ≥ 10
    ])
    b = np.array([6, 12])
    constraints = ['<=', '<=']
    
    lp = LinearProgramSolver(c, A, b, constraints, objective='max')
    solution, obj_value, status = lp.solve()
    
    print("Status:", status)
    print("Optimal Solution:", solution)
    print("Optimal Objective Value:", obj_value)
    
    # Example 2: Problem with unrestricted and non-positive variables
    print("\nExample 2: Unrestricted and Non-Positive Variables")
    c = np.array([2, -3])
    A = np.array([
        [1, 1],    # x + y ≤ 5
        [2, 1]     # 2x + y ≤ 10
    ])
    b = np.array([5, 10])
    constraints = ['<=', '<=']
    variable_types = ['unrestricted', 'non-positive']
    
    lp = LinearProgramSolver(c, A, b, constraints, variable_types, objective='min')
    solution, obj_value, status = lp.solve()
    
    print("Status:", status)
    print("Optimal Solution:", solution)
    print("Optimal Objective Value:", obj_value)

def main2():
    c = np.array([7, 2, 0, 6, 2, 4, 7,7, 6, 0, 2, 6, 5, 4,5, 4, 3, 4, 3, 6, 5,2, 4, 2, 3, 7, 5, 5,2, 4, 0, 6, 2, 5, 0,7, 2, 1, 2, 3, 1, 4,5, 1, 2, 3, 7, 2, 1])
    A = np.array([[0 for _ in range(49)] for _ in range(14)])
    b = np.array([10, 10, 10, 9, 10, 7, 9, 10, 10, 10, 10, 10, 7, 8])
    constraints = ['=' for _ in range(14)]
    for i in range(7):
        for j in range(7):
            A[i][i*7+j] = 1
    for i in range(7):
        for j in range(7):
            A[i+7][i+j*7] = 1
    
    lp = LinearProgramSolver(c, A, b, constraints, objective='min')
    solution, obj_value, status = lp.solve()
    print("Status:", status)
    print("Optimal Solution:", solution)
    print("Optimal Objective Value:", obj_value)


def main3():
    c = np.array([5, 3, 6, 7,5, 5, 7, 1,0, 0, 2, 3,5, 4, 6, 0])
    A = np.array([[0 for _ in range(16)] for _ in range(8)])
    b = np.array([9, 10, 10, 7, 8, 10, 9, 9])
    constraints = ['=' for _ in range(8)]
    for i in range(4):
        for j in range(4):
            A[i][i*4+j] = 1
    for i in range(4):
        for j in range(4):
            A[i+4][i+j*4] = 1
    
    lp = LinearProgramSolver(c, A, b, constraints, objective='min')
    solution, obj_value, status = lp.solve()
    print("Status:", status)
    print("Optimal Solution:", solution)
    print("Optimal Objective Value:", obj_value)


def example4():
    c = np.array([4, 3, 5, 5, 2, 6, 7, 4, 0])
    A = np.array([[0 for _ in range(9)] for _ in range(6)])
    b = np.array([5, 10, 7, 9, 6, 7])
    constraints = ['=' for _ in range(6)]
    for i in range(3):
        for j in range(3):
            A[i][i*3+j] = 1
    for i in range(3):
        for j in range(3):
            A[i+3][i+j*3] = 1
    
    lp = LinearProgramSolver(c, A, b, constraints, objective='min')
    solution, obj_value, status = lp.solve()
    print("Status:", status)
    print("Optimal Solution:", solution)
    print("Optimal Objective Value:", obj_value)



if __name__ == "__main__":
    example4()
