import numpy as np
from mealpy.swarm_based.HHO import OriginalHHO

def fitness_function(solution):
    return np.sum(solution**2)

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-10, -15, -4, -2, -8],
    "ub": [10, 15, 12, 8, 20],
    "minmax": "min",
}

epoch = 1000
pop_size = 50
model = OriginalHHO(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")