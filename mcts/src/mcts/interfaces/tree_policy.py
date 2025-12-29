from abc import ABC, abstractmethod


class TreePolicy[T, R](ABC):
    @abstractmethod
    def tree_policy(self, node: T) -> T:
        return node
    
    @abstractmethod
    def backpropagate(self, node: T, result: R) -> None:
        pass


