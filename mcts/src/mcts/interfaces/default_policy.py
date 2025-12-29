from abc import ABC, abstractmethod


class DefaultPolicy[T, R](ABC):
    @abstractmethod
    def simulate(self, node: T) -> R:
        pass