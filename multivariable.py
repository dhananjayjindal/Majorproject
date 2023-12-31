import numpy as np
from mealpy.optimizer import Optimizer

optimizer = Optimizer()

lowerbound = [0, 0]
upperbound = [50, 5]

T = 1000
N = 500
Dim = len(lowerbound)


def not_in_range(lowerbound, upperbound, x):
    if 2 * x[0] + 5 * x[1] > 98:
        return True

    for i in range(0, len(lowerbound)):
        if not lowerbound[i] <= x[i] <= upperbound[i]:
            return True

    return False


def fitness_function(x):
    ans = 2 * x[0] ** 2 - 7 * x[1] ** 2 + 12 * x[0] * x[1]
    return ans


population = np.transpose(
    [np.random.uniform(lowerbound[i], upperbound[i], N) for i in range(Dim)]
)

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

    i = 0
    while i != N:
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
                x_mean = np.mean([x for x in np.transpose(population)])
                pos_new = (best_rabbit_location - x_mean) - np.random.uniform() * (
                    np.array(lowerbound)
                    + np.random.uniform()
                    * (np.array(upperbound) - np.array(lowerbound))
                )

            if not_in_range(lowerbound, upperbound, pos_new):
                continue
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

                if not_in_range(lowerbound, upperbound, pos_new):
                    continue
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

                if not_in_range(lowerbound, upperbound, pos_new):
                    continue

                population_new.append(pos_new)
        i += 1
    population = population_new
    print(population_new)
    t += 1


print("best rabbit location = " + str(best_rabbit_location))
print("best rabbit fitness = " + str(best_rabbit_fitness))
print()
