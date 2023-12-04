import numpy as np
from mealpy.optimizer import Optimizer

T = 100
N = 10
optimizer = Optimizer()

lowerbound, upperbound = 0, 10


def fitness_function(coordinates):
    return -1 * coordinates**2 + 2 * coordinates + 11


population = np.random.uniform(lowerbound, upperbound, N)


t = 0

while t != T:
    
    # Initialize the best rabbit location (global best)
    best_rabbit_location = None
    best_rabbit_fitness = float("inf")
    fitness_values = [fitness_function(hawks) for hawks in population]
    best_hawk_index = np.argmax(fitness_values)

    if fitness_values[best_hawk_index] < best_rabbit_fitness:
        best_rabbit_location = population[best_hawk_index].copy()
        best_rabbit_fitness = fitness_values[best_hawk_index]

    population_new = []
    for i in range(0, N):
        E0 = 2 * np.random.uniform() - 1
        J0 = 2 * (1 - np.random.uniform())
        E = 2 * E0 * (1 - (t + 1) * 1.0 / T)

        # exploration phase
        if np.abs(E) >= 1:
            if np.random.rand() >= 0.5:  # perch based on other family members
                X_rand = population[np.random.randint(0, N)]
                pos_new = X_rand - np.random.uniform() * np.abs(
                    X_rand - 2 * np.random.uniform() * population[i]
                )
            else:
                x_mean = np.mean(population)
                pos_new = (best_rabbit_location - x_mean) - np.random.uniform() * (
                    lowerbound + np.random.uniform() * (upperbound - lowerbound)
                )
            population_new.append(pos_new)
        else:
            if np.random.rand() >= 0.5:
                if np.abs(E) >= 0.5:  # Soft besiege Eq. (6) in paper
                    pos_new = (
                        best_rabbit_location
                        - population[i]
                        - E * np.abs(J0 * best_rabbit_location - population[i])
                    )
                else:  # Hard besiege Eq. (4) in paper
                    pos_new = best_rabbit_location - E * np.abs(
                        best_rabbit_location - population[i]
                    )
                population_new.append(pos_new)

            else:
                LF_D = optimizer.get_levy_flight_step(
                    beta=1.5, multiplier=0.01, case=-1
                )
                if np.abs(E) >= 0.5:
                    Y = best_rabbit_location - E * np.abs(
                        J0 * best_rabbit_location - population[i]
                    )
                else:
                    x_mean = np.mean(population)
                    Y = best_rabbit_location - E * np.abs(
                        J0 * best_rabbit_location - x_mean
                    )

                S = np.random.uniform(lowerbound, upperbound)
                Z = Y + S * LF_D

                if fitness_function(Y) < fitness_values[i]:
                    pos_new = Y
                elif fitness_function(Z) < fitness_values[i]:
                    pos_new = Z
                else:
                    pos_new = population[i]

                population_new.append(pos_new)
                
    population = population_new
    t += 1


print(best_rabbit_fitness, best_rabbit_location)
