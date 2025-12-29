from collections.abc import Callable
from math import inf, log, sqrt
from random import Random

from mcts.interfaces import TreePolicy
from mcts.interfaces.state import State
from mcts.algorithms.uct.uct_tree_node import UctTreeNode
from typing import Any

class UctTreePolicy[A, S: State[Any, float, float]](TreePolicy[UctTreeNode[A, S], tuple[float, int]]):
    heuristic: Callable[[S], float] | None
    c_p: float
    c_h: float
    c_r: float
    random_gen: Random

    def __init__(self, heuristic: Callable[[S], float] | None = None, c_p = 1 / sqrt(2), c_h = 0.1, c_r = 0.0, seed: int | float | str | bytes | bytearray | None = None):
        self.heuristic = heuristic
        self.c_p = c_p
        self.c_h = c_h
        self.c_r = c_r
        self.random_gen = Random(seed)

    def compute_uct(self, n: int, node: UctTreeNode[A, S] | None) -> float:
        if node is None or node.N == 0:
            return inf
        
        exploit_term = node.W / node.N
        exploration_term = sqrt(log(n) / node.N)
        heuristic_term = self.heuristic(node.state) if self.heuristic != None else 0
        random_term = (self.random_gen.random() / n)

        return exploit_term + (self.c_p * exploration_term) + (self.c_h * heuristic_term) + (self.c_r * random_term)

    def tree_policy(self, node: UctTreeNode[A, S]) -> UctTreeNode[A, S]:
        if node.N == 0:
            return node
        
        next_node = self.expand(node)

        if next_node is None:
            next_node =  self.select(node)

        return next_node

    def select(self, node: UctTreeNode[A, S]) -> UctTreeNode[A, S]:
        if node.state.is_terminal():
            return node

        best_node = max(map(lambda n: (self.compute_uct(node.N, n), n), 
            node.children), key=lambda x: x[0])[1]
        
        return self.tree_policy(best_node)

    def expand(self, node: UctTreeNode[A, S]) -> UctTreeNode[A, S] | None:
        return node.expand()

    def backpropagate(self, node: UctTreeNode[A, S] | None, result: tuple[float, int]) -> None:
        while node != None:
            node.update(result)
            node = node.parent