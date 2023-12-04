import numpy as np
from mealpy.swarm_based.HHO import OriginalHHO

def fitness_function(solution):
    x1 = solution[0]
    x2 = solution[1]
    x3 = solution[2]
    return 10*(x1-1)**2+20*(x2-2)**2+30*(x3-3)**2

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [0,0,0],
    "ub": [10,10,10],
    "minmax": "max",
}

epoch = 1000
pop_size = 50
model = OriginalHHO(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")