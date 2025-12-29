from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Iterable

from mcts.interfaces import State
from enum import Enum
type GcpSolver = Callable[[Iterable[int], list[set[int]]], tuple[list[int], int]]

class MctsAlg(Enum):
    UCT = 0
    OPT = 1
    ALPHAGO = 2

class StateRep(Enum):
    SEQ = 0
    DSATUR = 1

@dataclass
class MctsConfig[S: State[Any, float, float]]:
    mcts_kind: MctsAlg
    state_rep: StateRep

    # MCTS
    max_iter: int
    max_time: int
    
    # Tree Policy
    heuristic_fun: Callable[[S], float] | None
    c_p: float | None
    c_h: float | None
    c_r: float | None

    # Default Policy
    n_simulations: int
    n_process: int | None

    # Other
    seed: int | float | str | bytes | bytearray | None

@dataclass
class BenchmarkResult:
    is_valid: bool
    coloring: list[int]
    time: float # Time elapsed in seconds
    n_iter: int
    V: int
    E: int
    best_sol: int

    @property
    def color_count(self):
        if not self.is_valid:
            return -1

        return max(self.coloring) + 1