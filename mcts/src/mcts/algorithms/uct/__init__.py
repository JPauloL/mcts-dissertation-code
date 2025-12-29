from mcts.algorithms.uct.uct_default_policy import UctDefaultPolicy
from mcts.algorithms.uct.uct_tree_node import UctTreeNode
from mcts.algorithms.uct.uct_tree_policy import UctTreePolicy
from mcts import MonteCarloTreeSearch

# class UctMcts(MonteCarloTreeSearch):
#     tree_policy: UctTreePolicy
#     default_policy: UctDefaultPolicy

#     def __init__(self, heuristic: Callable[[S], float] | None = None, c_p, c_h, c_r,
#             n_simulations: int = 1, n_procs: int = 1, 
#             seed: int | float | str | bytes | bytearray | None = None):
#         self.tree_policy = UctTreePolicy(heuristic, c_p, c_h, c_r, seed)
#         self.default_policy = UctDefaultPolicy(n_simulations, n_procs, seed)
#         super().__init__(self.tree_policy, self.default_policy)

__all__ = [
    "UctDefaultPolicy",
    "UctTreeNode",
    "UctTreePolicy"
]