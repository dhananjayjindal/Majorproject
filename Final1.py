import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHHO(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # -1 < E0 < 1
            E0 = 2 * np.random.uniform() - 1
            # factor to show the decreasing energy of rabbit
            E = 2 * E0 * (1 - (epoch + 1) * 1.0 / self.epoch)
            J = 2 * (1 - np.random.uniform())

            # -------- Exploration phase Eq. (1) in paper -------------------
            if np.abs(E) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                if np.random.rand() >= 0.5:  # perch based on other family members
                    X_rand = self.pop[np.random.randint(0, self.pop_size)][self.ID_POS].copy()
                    pos_new = X_rand - np.random.uniform() * np.abs(X_rand - 2 * np.random.uniform() * self.pop[idx][self.ID_POS])
                else:  # perch on a random tall tree (random site inside group's home range)
                    X_m = np.mean([x[self.ID_POS] for x in self.pop])
                    pos_new = (self.g_best[self.ID_POS] - X_m) - np.random.uniform() * \
                              (self.problem.lb + np.random.uniform() * (self.problem.ub - self.problem.lb))
                # pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                pop_new.append([pos_new, None])
            # -------- Exploitation phase -------------------
            else:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                if (np.random.rand() >= 0.5):
                    delta_X = self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]
                    if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                        pos_new = delta_X - E * np.abs(J * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:  # Soft besiege Eq. (4) in paper
                        pos_new = self.g_best[self.ID_POS] - E * np.abs(delta_X)
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    pop_new.append([pos_new, None])
                else:
                    LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                    if np.abs(E) >= 0.5:  # Soft besiege Eq. (10) in paper
                        Y = self.g_best[self.ID_POS] - E * np.abs(J * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:  # Hard besiege Eq. (11) in paper
                        X_m = np.mean([x[self.ID_POS] for x in self.pop])
                        Y = self.g_best[self.ID_POS] - E * np.abs(J * self.g_best[self.ID_POS] - X_m)
                    pos_Y = self.amend_position(Y, self.problem.lb, self.problem.ub)
                    target_Y = self.get_target_wrapper(pos_Y)
                    Z = Y + np.random.uniform(self.problem.lb, self.problem.ub) * LF_D
                    pos_Z = self.amend_position(Z, self.problem.lb, self.problem.ub)
                    target_Z = self.get_target_wrapper(pos_Z)
                    if self.compare_agent([pos_Y, target_Y], self.pop[idx]):
                        pop_new.append([pos_Y, target_Y])
                        continue
                    if self.compare_agent([pos_Z, target_Z], self.pop[idx]):
                        pop_new.append([pos_Z, target_Z])
                        continue
                    pop_new.append(self.pop[idx].copy())
        if self.mode not in self.AVAILABLE_MODES:
            for idx, agent in enumerate(pop_new):
                pop_new[idx][self.ID_TAR] = self.get_target_wrapper(agent[self.ID_POS])
        else:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        

def fitness_function(solution):
    # Decision variables
    x1, x2 = solution

    # List of constraint equations
    constraints = [
        # Constraint 1: 2x1 + x2 <= 10
        # Constraint 2: x1 + 3x2 <= 12
        x1 + x2 <= 4, 
        x1 >= 0, 
        x2 >= 0
        # Add more constraints here...
    ]

    # List to store constraint penalties
    penalties = [max(0, constraint) for constraint in constraints]

    # Calculate the total penalty for constraint violations
    total_penalty = sum(penalties)

    # Define the objective function (you can change this)
    objective_value =  2*x1 + 3*x2

    # Calculate the fitness value with constraints
    fitness_with_constraints = objective_value - total_penalty

    return fitness_with_constraints

# ...


problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [0,0],
    "ub": [500,500],
    "minmax": "max",
}

epoch = 10
pop_size = 50

model = OriginalHHO(epoch, pop_size)
best_position, best_fitness = model.solve(problem_dict1)

print(f"Solution: {best_position}, Fitness: {best_fitness}")