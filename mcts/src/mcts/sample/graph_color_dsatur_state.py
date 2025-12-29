from collections.abc import Iterator
from typing import cast
from mcts.interfaces import State

class GraphColorDsaturActionIterator:
    _vertex: int = -1
    _color: int = 0 # Index for iterator
    _neighbors_colors: set[int]
    _is_terminal: bool = False

    def __init__(self, state: GraphColorDsaturState, is_default: bool = False) -> None:
        self.state = state

        if state.is_terminal():
            self._is_terminal = True
            return
        
        max_colors = -1

        for node in range(len(self.state.graph)):
            if node in self.state.colorings.keys():
                continue

            neighbors = self.state.graph[node]
            colors = self.state.neighbors_colors[node] if node in self.state.neighbors_colors else set()

            if len(colors) > max_colors or (len(colors) == max_colors and len(neighbors) > len(self.state.graph[self._vertex])):
                max_colors = len(colors)
                self._vertex = node

            if is_default:
                break

        self._neighbors_colors = self.state.neighbors_colors[self._vertex] if self._vertex != -1 and self._vertex in self.state.neighbors_colors else set()
        self._possible_colors_count = state.n_colors - len(self._neighbors_colors) + 1
        self._weight_total = self._possible_colors_count

    def __iter__(self):
        self._color = 0
        return self

    def __next__(self) -> tuple[int, int]:
        while not self._is_terminal and self._color <= self.state.n_colors:
            new_assignment = (self._vertex, self._color)
            self._color += 1

            if new_assignment[1] in self._neighbors_colors:
                continue

            return new_assignment

        if self._is_terminal:
            print(self.state.coloring)
            exit()

        raise StopIteration


class GraphColorDsaturState(State[tuple[int, int], float, float]):
    graph: list[set[int]]
    colorings: dict[int, int]
    neighbors_colors: dict[int, set[int]]
    n_colors: int

    _hash: int | None = None

    @property
    def coloring(self):
        return [self.colorings[v] for v in range(len(self.colorings))]

    def __init__(self, graph: list[set[int]], colorings: dict[int, int] = {}, neighbors_colors: dict[int, set[int]] = {}, n_colors: int = 0) -> None:
        self.graph = graph
        self.colorings = colorings
        self.neighbors_colors = neighbors_colors
        self.n_colors = n_colors

    def is_terminal(self) -> bool:
        return len(self.colorings) == len(self.graph)

    def interpret_reward(self, reward: float) -> float:
        return reward

    def get_reward(self) -> float:
        return -self.n_colors

    def actions_tree(self) -> Iterator[tuple[int, int]]:
        return GraphColorDsaturActionIterator(self)
    
    def actions_default(self) -> Iterator[tuple[int, int]]:
        return GraphColorDsaturActionIterator(self)
        # return GraphColorDsaturActionIterator(self, is_default=True, is_weighted=True)

    def play(self, action: tuple[int, int]) -> GraphColorDsaturState:
        new_state = self.colorings.copy()
        new_state[action[0]] = action[1]

        neighbors_colors = self.neighbors_colors.copy()

        for neighbor in self.graph[action[0]]:
            if neighbor in self.colorings:
                continue

            neighbors_colors[neighbor] = neighbors_colors[neighbor].copy() if neighbor in neighbors_colors else set()
            neighbors_colors[neighbor].add(action[1])

        return GraphColorDsaturState(self.graph, new_state, neighbors_colors, max(self.n_colors, action[1] + 1))

    def equals(self, other: GraphColorDsaturState) -> bool:
        return self.colorings == other.colorings

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        else:
            return hash(tuple(self.colorings.items()))
            
    def __eq__(self, other: object) -> bool:
        try:
            other_state = cast(GraphColorDsaturState, other)
            return self.equals(other_state)
        except:
            return False

    def is_valid(self) -> bool:
        for index, color in self.colorings.items():
            for neighbor in self.graph[index]:
                if neighbor in self.colorings and self.colorings[neighbor] == color:
                    return False
                
        return True