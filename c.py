import numpy as np

def hho_algorithm(objective_function, dimension, search_space, num_iterations, num_hawks):
    # Initialize hawks' positions and velocities randomly within the search space
    positions = np.random.uniform(search_space[0], search_space[1], (num_hawks, dimension))
    velocities = np.random.uniform(-1, 1, (num_hawks, dimension))
    
    for i in range(num_iterations):
        for j in range(num_hawks):
            # Evaluate the fitness of each hawk's position
            fitness = objective_function(positions[j])
            
            # Update the global best position
            if j == 0 or fitness < best_fitness:
                best_fitness = fitness
                best_position = positions[j].copy()
                
            # Explore and exploit using Harris' hawks strategy
            exploration_prob = np.random.random()
            if exploration_prob < 0.5:
                # Exploration: Move towards the best position
                velocities[j] = velocities[j] + np.random.random() * (best_position - positions[j])
            else:
                # Exploitation: Move towards a random hawk's position
                random_hawk = np.random.randint(0, num_hawks)
                velocities[j] = velocities[j] + np.random.random() * (positions[random_hawk] - positions[j])
            
            # Update the hawk's position using the calculated velocity
            positions[j] = positions[j] + velocities[j]
            
            # Ensure the positions are within the search space
            positions[j] = np.clip(positions[j], search_space[0], search_space[1])
    
    return best_position, best_fitness

# Example objective function (you should replace this with your own)
def objective_function(x):
    return np.sum(x**2)

if __name__ == "__main__":
    # Define the problem dimensions and search space
    dimension = 10
    search_space = (-10, 10)  # Search space for each dimension
    
    # Set the algorithm parameters
    num_iterations = 100
    num_hawks = 20
    
    # Run the HHO algorithm
    best_position, best_fitness = hho_algorithm(objective_function, dimension, search_space, num_iterations, num_hawks)
    
    # Print the results
    print("Best position:", best_position)
    print("Best fitness:", best_fitness)
