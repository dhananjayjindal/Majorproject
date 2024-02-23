import numpy as np
from mealpy.optimizer import Optimizer

optimizer = Optimizer()

lowerbound = [0, 0]
upperbound = [5, 5]

T = 1000
N = 1000
# optimum_given = float("inf")
Dim = len(lowerbound)


def franctional_constraints(a, b, tpe, value):
    if b == 0:
        return False
    if tpe == "le":
        return (a / b) > value
    if tpe == "ge":
        return (a / b) < value
    if tpe == "g":
        return (a / b) <= value
    if tpe == "l":
        return (a / b) >= value


def not_in_range(lowerbound, upperbound, x):
    c_1_n = 2 * x[0] + x[1]
    c_1_d = 1
    c_2_n = x[0] + 3 * x[1]
    c_2_d = 1
    # c_3_n = x[0] + x[1] + x[2]
    # c_3_d = 1
    if (
        franctional_constraints(c_1_n, c_1_d, "ge", 6)
        or franctional_constraints(c_2_n, c_2_d, "ge", 8)
        # or franctional_constraints(c_3_n, c_3_d, "le", 1.01)
    ):
        return True

    for i in range(0, len(lowerbound)):
        if not lowerbound[i] <= x[i] <= upperbound[i]:
            return True

    return False


def fitness_function(x):
    # a = x.T*Σ*x
    a = (2 * x[0] + x[1]) ** 2 + 1
    b = x[0] + 2 * x[1] + 1
    if b == 0:
        return False
    return -a / b


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

                fitness_of_Y = fitness_function(Y)
                fitness_of_Z = fitness_function(Z)
                if not fitness_of_Y or not fitness_of_Z:
                    continue

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
    # if np.abs(best_rabbit_fitness - optimum_given) < 0.001:
    #     break
    print(t)
    t += 1

formatted_best_rabbit_location = []
for e in best_rabbit_location:
    formatted_result = "{:.3f}".format(e)
    formatted_best_rabbit_location.append(formatted_result)

formatted_best_rabbit_fitness = "{:.3f}".format(best_rabbit_fitness)
print("*********************************************************************")
print("best rabbit location = " + str(formatted_best_rabbit_location))
print("best rabbit fitness = " + str(formatted_best_rabbit_fitness))


# # Assuming result is your NumPy array
# result = np.array([formatted_best_rabbit_location])

# # Save the array to a file
# np.save('result.npy', result)
