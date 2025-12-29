from collections.abc import Iterator
from typing import cast
from mcts.interfaces import State

class GraphColorSeqActionIterator:
    _color: int # Index for iterator
    _neighbors_colors: set[int]
    _is_terminal = False

    def __init__(self, state: GraphColorSeqState) -> None:
        self.state = state
        self._color = 0
        self._neighbors_colors = set()

        if state.is_terminal():
            self._is_terminal = True
            return

        for adj in state.graph[len(state.colorings)]:
            if adj < len(state.colorings):
                self._neighbors_colors.add(state.colorings[adj])

        self._possible_colors_count = state.n_colors - len(self._neighbors_colors) + 1

    def heuristic(self, color: int) -> float:
        return 1 - (color / (self._possible_colors_count))

    def __iter__(self):
        self._color = 0
        return self

    def __next__(self) -> tuple[int, float]:
        while not self._is_terminal and self._color <= self.state.n_colors:
            new_color = self._color
            self._color += 1

            if new_color in self._neighbors_colors:
                continue

            return (new_color, self.heuristic(new_color))

        raise StopIteration

class GraphColorSeqState(State[tuple[int, float], float, float]):
    graph: list[set[int]]
    colorings: tuple[int, ...]
    n_colors: int
    heuristic_value: float

    _hash: int | None = None

    @property
    def coloring(self):
        return self.colorings

    def __init__(self, graph: list[set[int]], colorings: tuple[int, ...] = (), n_colors: int = 0, heuristic_value: float = 0) -> None:
        self.graph = graph
        self.colorings = colorings
        self.n_colors = n_colors
        self.heuristic_value = heuristic_value

    def is_terminal(self) -> bool:
        return len(self.colorings) == len(self.graph)

    def interpret_reward(self, reward: float) -> float:
        return reward

    def get_reward(self) -> float:
        return -self.n_colors

    def actions_tree(self) -> Iterator[tuple[int, float]]:
        return GraphColorSeqActionIterator(self)
    
    def actions_default(self) -> Iterator[tuple[int, float]]:
        return GraphColorSeqActionIterator(self)

    def play(self, action: tuple[int, float]) -> GraphColorSeqState:
        new_state = self.colorings + (action[0],)
        
        return GraphColorSeqState(self.graph, new_state, max(self.n_colors, action[0] + 1), action[1])

    def equals(self, other: GraphColorSeqState) -> bool:
        return self.colorings == other.colorings

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        else:
            return hash(self.colorings)
    
    def __eq__(self, other: object) -> bool:
        try:
            other_state = cast(GraphColorSeqState, other)
            return self.equals(other_state)
        except:
            return False
        
    def is_valid(self) -> bool:
        for index, color in enumerate(self.colorings):
            for neighbor in self.graph[index]:
                if neighbor < len(self.colorings) and self.colorings[neighbor] == color:
                    return False
                
        return True