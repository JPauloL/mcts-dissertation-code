from math import inf
from random import Random

from mcts.monte_carlo_tree_search import DefaultPolicy
from mcts.algorithms.opt.opt_tree_node import OptTreeNode
from mcts.interfaces import State
from typing import Any

class OptDefaultPolicy[A, S: State[Any, float, float]](DefaultPolicy[OptTreeNode[A, S], tuple[float, int]]):
    random_gen: Random
    best_solution: tuple[S | None, float]

    def __init__(self, seed: int | float | str | bytes | bytearray | None = None) -> None:
        self.best_solution = (None, -inf)
        self.random_gen = Random(seed)

    def simulate(self, node: OptTreeNode[A, S]) -> tuple[float, int]:
        cur_state = node.state
        
        while not cur_state.is_terminal():
            children = list(cur_state.actions_default())
            action = self.random_gen.choice(children)
            cur_state = cur_state.play(action)

        reward = cur_state.get_reward()

        if cur_state.interpret_reward(reward) > self.best_solution[1]:
            self.best_solution = (cur_state, cur_state.interpret_reward(reward))
            

        return (reward, 1)