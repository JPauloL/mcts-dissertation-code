from typing import Any, Iterable, cast
from mcts.algorithms.ag.ag_tree_node import AgTreeNode
from mcts.algorithms.opt import OptDefaultPolicy, OptTreeNode, OptTreePolicy
from mcts.algorithms.uct import UctDefaultPolicy, UctTreeNode, UctTreePolicy
from mcts import MonteCarloTreeSearch
from mcts.interfaces.state import State
from mcts.other.solutions import Solutions
from mcts.sample.graph_color_dsatur_state import GraphColorDsaturState
from mcts.sample.graph_color_seq_state import GraphColorSeqState

from src.types.types import MctsConfig, StateRep

def state_from_kind(kind: StateRep, graph: list[set[int]]) -> GraphColorDsaturState | GraphColorSeqState:
    if kind == StateRep.DSATUR:
        return GraphColorDsaturState(graph)
    else:
        return GraphColorSeqState(graph)

def mcts_from_config(config: MctsConfig) -> MonteCarloTreeSearch:
    tree_policy = UctTreePolicy(lambda x: cast(GraphColorSeqState, x).heuristic_value, seed=config.seed)
    default_policy = UctDefaultPolicy(seed=config.seed)

    mcts = MonteCarloTreeSearch(tree_policy, default_policy)

    return mcts

def node_from_config(config: MctsConfig, graph) -> Any:
    return UctTreeNode(graph)

# def run_mcts(config: MctsConfig, graph: list[set[int]]) -> tuple[list[int], int]:

def default_policy_weights(state: GraphColorDsaturState, actions: list[tuple[int, int]]) -> list[float]:
    weights = [1.0]

    for action in actions[1:]:
        weights.append(0)
    return weights

def tree_policy_heuristic(state: GraphColorDsaturState, actions: list[tuple[int, int]]) -> float:
    return 1.0

def run_uct_mcts(config: MctsConfig, graph: list[set[int]]) -> tuple[list[int], int]:
    solutions = Solutions()
    tree_policy = UctTreePolicy(seed=config.seed)
    default_policy = UctDefaultPolicy(heuristic=default_policy_weights, solutions=solutions, seed=config.seed)

    state = state_from_kind(config.state_rep, graph)
    solver = mcts_from_config(config)
    root = node_from_config(config, state)

    solver = MonteCarloTreeSearch(tree_policy, default_policy)

    solver.run(root, config.max_iter)

    best_solution = solutions.best_solution

    return (best_solution[0].coloring if best_solution != None else [], cast(int, -best_solution[1] if best_solution is not None else -1))

def run_opt_mcts(seed, n_iter: int, graph: list[set[int]]) -> tuple[list[int], int]:
    tree_policy = OptTreePolicy(seed=seed)
    default_policy = OptDefaultPolicy(seed=seed)

    state = GraphColorSeqState(graph)
    root = OptTreeNode(state)

    solver = MonteCarloTreeSearch(tree_policy, default_policy)

    solver.run(root, n_iter)

    best_solution = default_policy.best_solution

    return (best_solution[0].coloring if best_solution[0] is not None else [], cast(int, -best_solution[1]))