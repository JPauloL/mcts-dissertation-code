from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Hashable, Self 


class State[A, R1, R2](ABC, Hashable):
    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def interpret_reward(self, reward: R1) -> R2:
        pass

    @abstractmethod
    def get_reward(self) -> R1:
        pass

    @abstractmethod
    def actions_tree(self) -> Iterator[A]:
        pass

    @abstractmethod
    def actions_default(self) -> Iterator[A]:
        pass
    
    @abstractmethod
    def play(self, action: A) -> Self:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass