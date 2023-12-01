import numpy as np

# Define the objective function to be optimized (quadratic equation)
def objective_function(x):
    return 50 * x[0] + 100 * x[1]

# Define the fitness function for HHO (negative of the objective function)
def fitness_function(solution):
    return objective_function(solution)

# Define the bounds of the search space
lb = np.array([0, 0])  # Lower bounds
ub = np.array([50, 50])    # Upper bounds

# Set HHO parameters
population_size = 50
max_iterations = 1000

# Initialize the population of hawks randomly within the search space
population = np.random.uniform(lb, ub, (population_size, len(lb)))
print(population)
# Main optimization loop
for iteration in range(max_iterations):
    # Calculate fitness value for each hawk
    fitness_values = [fitness_function(hawk) for hawk in population]

    # Find the index of the best hawk (prey)
    best_hawk_index = np.argmax(fitness_values)

    # Update the best solution found
    best_solution = population[best_hawk_index]
    best_fitness = fitness_values[best_hawk_index]

    # Iterate through each hawk
    for i in range(population_size):
        # Calculate energy (E) and a random number (r) for each hawk
        E = np.random.rand()
        r = np.random.rand()

        # Update the hawk's location using different strategies
        if E >= 0.5 and r >= 0.5:  # Soft Round-Up
            population[i] += 0.1 * np.random.uniform(lb, ub, len(lb))
        elif E < 0.5 and r >= 0.5:  # Hard Round-Up
            population[i] += 0.2 * np.random.uniform(lb, ub, len(lb))
        elif E >= 0.5 and r < 0.5:  # Soft Round-Up with progressive rapid dives
            population[i] += 0.3 * np.random.uniform(lb, ub, len(lb))
        elif E < 0.5 and r < 0.5:  # Hard Round-Up with progressive rapid dives
            population[i] += 0.4 * np.random.uniform(lb, ub, len(lb))

    # Clip the hawk's positions to stay within the bounds
    population = np.clip(population, lb, ub)

# Print the results
print("Optimization Result:")
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
