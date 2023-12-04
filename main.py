import numpy as np

# Define the fitness function to be optimized (replace with your own)
def fitness_function(x):
    print(x)
    x1 = x[0]
    x2 = x[1]
    # This is a sample fitness function; replace it with your own objective function.
    return 9.82*x1*x2 - 2*x1*x1  # Example: minimize the sum of squares

# Initialize parameters
population_size = 20
max_iterations = 100
search_space_dim = 2  # Adjust based on your problem's dimensionality
alpha = 0.1  # Exploration coefficient
beta = 1.5   # Exploitation coefficient

# Initialize the population of hawks
population = np.random.rand(population_size, search_space_dim)

# Main optimization loop
for iteration in range(max_iterations):
    # Evaluate fitness for each hawk
    fitness_values = [fitness_function(hawk) for hawk in population]
    
    # Find the best (prey) hawk
    best_hawk_index = np.argmin(fitness_values)
    best_hawk = population[best_hawk_index]
    
    # Exploration phase (update positions of hawks)
    for i in range(population_size):
        if i != best_hawk_index:
            exploration = alpha * np.random.rand(search_space_dim)
            population[i] = population[i] + exploration
    
    # Exploitation phase (update positions towards the best hawk)
    for i in range(population_size):
        if i != best_hawk_index:
            exploitation = beta * (best_hawk - population[i])
            population[i] = population[i] + exploitation
    
    # Apply any constraints if needed (e.g., boundary constraints)
    # population[i] = apply_constraints(population[i])
    
    # Print the best fitness value at each iteration
    best_fitness = min(fitness_values)
    print(f"Iteration {iteration}: Best Fitness = {best_fitness}")
    
# Output the best solution found
final_best_solution = population[best_hawk_index]
final_best_fitness = fitness_values[best_hawk_index]
print("\nOptimization Result:")
print(f"Best Solution: {final_best_solution}")
print(f"Best Fitness: {final_best_fitness}")
