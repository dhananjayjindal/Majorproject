import numpy as np

# Define the fitness function to be minimized (replace with your own)
def fitness_function(x):
    # Example: Minimize the sum of squares
    return sum(x**2)

# Define HHO parameters
population_size = 20
max_iterations = 100
search_space_dim = 5  # Adjust based on your problem's dimensionality

# Initialize the population of hawks randomly
population = np.random.rand(population_size, search_space_dim)

# Initialize energy and jump strength for each hawk
energy = np.random.rand(population_size)
jump_strength = np.random.rand(population_size)

# Initialize the best rabbit location (global best)
best_rabbit_location = None
best_rabbit_fitness = float('inf')

# Main optimization loop
for iteration in range(max_iterations):
    # Calculate fitness value for each hawk
    fitness_values = [fitness_function(hawk) for hawk in population]
    
    # Find the index of the best hawk (prey)
    best_hawk_index = np.argmin(fitness_values)
    
    # Update the best rabbit location if needed
    if fitness_values[best_hawk_index] < best_rabbit_fitness:
        best_rabbit_location = population[best_hawk_index].copy()
        best_rabbit_fitness = fitness_values[best_hawk_index]
    
    # Iterate through each hawk
    for i in range(population_size):
        # Calculate energy and a random number for each hawk
        E = energy[i]
        r = np.random.rand()
        
        # Update the hawk's location based on HHO strategies
        if E >= 0.5 and r >= 0.5:  # Soft Round-Up
            population[i] += 0.1 * np.random.rand(search_space_dim)
        elif E < 0.5 and r >= 0.5:  # Hard Round-Up
            population[i] += 0.2 * np.random.rand(search_space_dim)
        elif E >= 0.5 and r < 0.5:  # Soft Round-Up with progressive rapid dives
            population[i] += 0.3 * np.random.rand(search_space_dim)
        elif E < 0.5 and r < 0.5:  # Hard Round-Up with progressive rapid dives
            population[i] += 0.4 * np.random.rand(search_space_dim)
        
        # Apply constraints if needed (e.g., boundary constraints)
        # population[i] = apply_constraints(population[i])
        
        # Update energy and jump strength for the hawk
        energy[i] = 0.9 * energy[i]
        jump_strength[i] = 0.8 * jump_strength[i]
    
# Output the best solution found
print("\nOptimization Result:")
print(f"Best Rabbit Location: {best_rabbit_location}")
print(f"Best Rabbit Fitness: {best_rabbit_fitness}")
