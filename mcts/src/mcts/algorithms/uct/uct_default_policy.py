from collections.abc import Callable
from math import inf
from random import Random
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from typing import Any

from mcts.algorithms.ag.ag_tree_node import AgTreeNode
from mcts.algorithms.opt.opt_tree_node import OptTreeNode
from mcts.monte_carlo_tree_search import DefaultPolicy
from mcts.interfaces import State
from mcts.algorithms.uct.uct_tree_node import UctTreeNode
from mcts.other.solutions import Solutions

class UctDefaultPolicy[A, S: State[Any, float, float]](DefaultPolicy[UctTreeNode[A, S] | OptTreeNode[A, S] | AgTreeNode[A, S], tuple[float, int]]):
    heur: Callable[[S, list[A]], list[float]] | None
    random_gen: Random
    solutions: Solutions | None
    pool: Pool | None

    def __init__(self, n_sims = 1, n_threads = 1, heuristic: Callable[[S, list[A]], list[float]] | None = None, solutions: Solutions | None = None, seed: int | float | str | bytes | bytearray | None = None) -> None:
        self.best_solution = (None, -inf)
        self.random_gen = Random(seed)
        self.n_sims = n_sims
        self.heur = heuristic
        self.solutions = solutions
        self.pool = Pool(n_threads) if n_threads > 1 else None

    @staticmethod
    def run_simulation(node: UctTreeNode[A, S] | OptTreeNode[A, S] | AgTreeNode[A, S], heur: Callable[[S, list[A]], list[float]] | None, random_gen: Random) -> tuple[S, float]:
        cur_state = node.state
        
        while not cur_state.is_terminal():
            actions = [a for a in cur_state.actions_default()]
            weights = heur(node.state, actions) if heur is not None else None

            action = random_gen.choices(actions, weights)[0]
            cur_state = cur_state.play(action)

        reward = cur_state.get_reward()

        return cur_state, reward

    def simulate(self, node: UctTreeNode[A, S] | OptTreeNode[A, S] | AgTreeNode[A, S]) -> tuple[float, int]:
        if self.pool is not None:
            results = self.pool.starmap(self.run_simulation, [(node, self.heur, self.random_gen)] * self.n_sims)
        else:
            results = [self.run_simulation(node, self.heur, self.random_gen) for _ in range(self.n_sims)]
    
        if self.solutions is not None:
            self.solutions.add_solutions(results)

        return (sum(res[1] for res in results), self.n_sims)