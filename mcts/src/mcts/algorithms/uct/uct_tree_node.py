from collections.abc import Iterator
from mcts.interfaces import State
from typing import Any


class UctTreeNode[A, S: State[Any, float, float]]:
    state: S # State of the node
    N: int # Number of times the node has been visited
    W: float # Total reward
    parent: UctTreeNode[A, S] | None # Parent of the node or None if this node is the root
    children: list[UctTreeNode[A, S]]

    _children_iter: Iterator[A]

    def __init__(self, state: S, parent: UctTreeNode[A, S] | None = None):
        self.state = state
        self.N = 0
        self.W = 0 
        self.parent = parent
        self.children = []
        self._children_iter = state.actions_tree()

    def update(self, result: tuple[float, int]) -> None:
        self.N += result[1]
        self.W += self.state.interpret_reward(result[0])

    def add_child(self, state: S) -> UctTreeNode[A, S]:
        new_node = UctTreeNode(state, self)
        self.children.append(new_node)
        return new_node
    
    def expand(self) -> UctTreeNode[A, S] | None:
        try:
            action = next(self._children_iter)
            new_state = self.state.play(action)
            return self.add_child(new_state)
        except StopIteration:
            return None

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def estimated_value(self):
        return self.W / self.N if self.N > 0 else 0

    def best_child(self) -> UctTreeNode[A, S]:
        best_child = self.children[0]
        best_value = best_child.estimated_value()

        for child in self.children[:-1]:
            if child.estimated_value() > best_value:
                best_child = child

        return best_child
        
    def policy_array(self) -> list[tuple[UctTreeNode[A, S], float]]:
        policy_array = list(map(lambda node: (node, node.estimated_value()), self.children))
        return policy_array
    
    def count(self) -> int:
        return 1 + sum(map(lambda x: x.count(), self.children))