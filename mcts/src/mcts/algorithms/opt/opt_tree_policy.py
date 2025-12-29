from math import inf, log, sqrt
from random import Random
from typing import Any
from mcts.interfaces import TreePolicy
from mcts.algorithms.opt.opt_tree_node import OptTreeNode
from mcts.interfaces import State

class OptTreePolicy[A, S: State[Any, float, float]](TreePolicy[OptTreeNode[A, S], tuple[float, int]]):
    c_p: float
    c_h: float
    c_r: float
    random_gen: Random

    def __init__(self, c_p = 1 / sqrt(2), c_h = 0.1, c_r = 0.0, seed: int | float | str | bytes | bytearray | None = None):
        self.c_p = c_p
        self.c_h = c_h
        self.c_r = c_r
        self.random_gen = Random(seed)

    def compute_uct(self, parent: OptTreeNode[A, S], node: OptTreeNode[A, S]) -> float:
        if parent.best_solution == parent.worst_solution:
            return 0
        
        exploit_term = (node.best_solution - parent.worst_solution) / (parent.best_solution - parent.worst_solution)
        exploration_term = sqrt(2.0 * log(parent.N) / node.N)
        heuristic_term = 0
        random_term = (self.random_gen.random() / parent.N)

        return exploit_term + (self.c_p * exploration_term) + (self.c_h * heuristic_term) + (self.c_r * random_term)

    def tree_policy(self, node: OptTreeNode[A, S]) -> OptTreeNode[A, S]:
        if node.N == 0:
            return node
        
        next_node = self.expand(node)

        if next_node is None:
            next_node =  self.select(node)

        return next_node

    def select(self, node: OptTreeNode[A, S]) -> OptTreeNode[A, S]:
        if node.state.is_terminal():
            return node

        best_node = max(map(lambda n: (self.compute_uct(node, n[1]), n[1]), 
            node.children), key=lambda x: x[0])[1]
        
        return self.tree_policy(best_node)

    def expand(self, node: OptTreeNode[A, S]) -> OptTreeNode[A, S] | None:
        return node.expand()

    def backpropagate(self, node: OptTreeNode[A, S] | None, result: tuple[float, int]) -> None:
        while node != None:
            node.update(result)
            node = node.parent