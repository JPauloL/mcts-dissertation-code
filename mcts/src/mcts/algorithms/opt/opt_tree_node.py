from collections.abc import Iterator
from math import inf
from typing import Any

from mcts.interfaces import State

class OptTreeNode[A, S: State[Any, float, float]]:
    state: S # State of the node
    N: float
    best_solution: float # 
    worst_solution: float # 
    parent: OptTreeNode[A, S] | None # Parent of the node or None if this node is the root
    children: list[tuple[A, OptTreeNode[A, S]]]

    _children_iter: Iterator[A]

    def __init__(self, state: S, parent: OptTreeNode[A, S] | None = None):
        self.state = state
        self.best_solution = -inf
        self.worst_solution = inf
        self.N = 0
        self.parent = parent
        self.children = []
        self._children_iter = state.actions_tree()

    def update(self, result: tuple[float, int]) -> None:
        self.N += result[1]
        value = self.state.interpret_reward(result[0])

        if value > self.best_solution:
            self.best_solution = value
        
        if value < self.worst_solution:
            self.worst_solution = value

    def add_child(self, action: A) -> OptTreeNode[A, S]:
        new_state = self.state.play(action)
        new_node = OptTreeNode(new_state, self)
        self.children.append((action, new_node))
        return new_node
    
    def expand(self) -> OptTreeNode[A, S] | None:
        try:
            action = next(self._children_iter)
            return self.add_child(action)
        except StopIteration:
            return None

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def estimated_value(self):
        return self.best_solution

    def best_child(self) -> tuple[A, OptTreeNode[A, S]]:
        best_child = self.children[0]
        best_value = best_child[1].estimated_value()

        for child in self.children[:-1]:
            if child[1].estimated_value() > best_value:
                best_child = child

        return best_child
        
    def policy_array(self) -> list[tuple[OptTreeNode[A, S], float]]:
        policy_array = list(map(lambda node: (node[1], node[1].estimated_value()), self.children))
        return policy_array
    
    def count(self) -> int:
        return 1 + sum(child[1].count() for child in self.children)