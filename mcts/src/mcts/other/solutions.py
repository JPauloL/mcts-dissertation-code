type Solution[A] = tuple[A, float]

class Solutions[A]:
    keep_history: bool
    history: list[Solution[A]]
    best_solution: Solution[A] | None
    
    def __init__(self, keep_history: bool = False):
        self.keep_history = keep_history
        self.history = []
        self.best_solution = None

    def add_solution(self, solution: Solution[A]) -> None:
        if self.best_solution is None or solution[1] > self.best_solution[1]:
            self.best_solution = solution

        if self.keep_history:
            self.history.append(solution)

    def add_solutions(self, solutions: list[Solution[A]]) -> None:
        for solution in solutions:
            self.add_solution(solution)