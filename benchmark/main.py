from dataclasses import dataclass
from time import time
from typing import Iterable
import matplotlib.pyplot as plt
from functools import partial
from collections.abc import Callable
from multiprocessing import Pool
from random import Random

from src.types import BenchmarkResult, GcpSolver, MctsConfig
from src.algorithms.mcts_setups import run_opt_mcts, run_uct_mcts
from src.utils import read_graph
from src.algorithms import dsatur, largest_first, recursive_largest_fit, seq_assignment
from config import all_graphs, all_ub_graphs, dimacs_solved_graphs, all_solved_graphs, mcts_config, n_proc, seed, BenchmarkGraph



loaded_graphs: dict[str, tuple[range, list[set[int]]]] = {}

def check_coloring(edges: list[set[int]], coloring: list[int], complete: bool = True) -> bool:
    if complete and len(coloring) != len(edges):
        return False

    for index, color in enumerate(coloring):
        for neighbor in edges[index]:
            if neighbor < len(coloring) and coloring[neighbor] == color:
                return False
            
    return True

def test_heuristic(graphs: list[BenchmarkGraph], heur: GcpSolver) -> list[BenchmarkResult]:
    # x = []
    # y = []
    # abs_sum = 0
    # square_sum = 0

    benchmark_results = []

    for (path, (n, e), best_sol, orig) in graphs:
        nodes, adj = loaded_graphs[path] if path in loaded_graphs else read_graph(f"instances/{path}")
        t0 = time()
        colors, k = heur(nodes, adj)
        total_time = time() - t0

        is_valid = check_coloring(adj, colors)
        res = BenchmarkResult(is_valid, colors, total_time, -1, n, e, best_sol)

        benchmark_results.append(res)

    return benchmark_results

def run_all_graphs(solver: GcpSolver) -> list[BenchmarkResult]:
    return test_heuristic(all_solved_graphs, solver)

def run_all_solved_graphs(solver: GcpSolver) -> list[BenchmarkResult]:
    return test_heuristic(all_solved_graphs, solver)

def run_all_ub_graphs(solver: GcpSolver) -> list[BenchmarkResult]:
    return test_heuristic(all_solved_graphs, solver)

def run_all_unsolved_graphs(solver: GcpSolver) -> list[BenchmarkResult]:
    return test_heuristic(all_solved_graphs, solver)

def run_benchmark(graphs: list[BenchmarkGraph], solver: GcpSolver):
    return test_heuristic(graphs, solver)

def print_result(alg_name: str, results: list[BenchmarkResult], verbose=True) -> None:
    print(f"{alg_name}:")

    if verbose:
        for res in results:
            print("Found", res.color_count, "with a known best of", res.best_sol, "in", res.time, "seconds")

    abs_sum = sum((res.color_count - res.best_sol) for res in results if res.best_sol != -1)
    sqr_sum = sum((res.color_count - res.best_sol) ** 2 for res in results if res.best_sol != -1)

    print("Mean Absolute Error (MAE):", abs_sum / len(results))
    print("Mean Square Error (MSE):", sqr_sum / len(results))

def mcts_name_from_config(config: MctsConfig):
    return f"MCTS {config.mcts_kind} {config.max_iter}I {"HEUR" if config.heuristic_fun is None else ""} {config.n_simulations}SIM"

def main():
    random_gen = Random(seed)
    pool = Pool(n_proc)

    # graph_sample = random_gen.sample(all_solved_graphs, 5)
    graph_sample = all_ub_graphs

    heur_algs = [
        seq_assignment,
        largest_first,
        recursive_largest_fit,
        dsatur
    ]

    t0 = time()

    heur_res = pool.starmap(run_benchmark, ((graph_sample, heur) for heur in heur_algs))
    pool.close()

    mcts_res: dict[str, list[BenchmarkResult]] = {}
    for mcts_input in mcts_config:
        mcts_res[mcts_name_from_config(mcts_input)] = run_benchmark(graph_sample, lambda x, y: run_uct_mcts(mcts_input, y))

    total_time = time() - t0

    print_result("Sequential Assignment", heur_res[0], False)
    print_result("Largest First", heur_res[1], False)
    print_result("Recursive Largest Fit", heur_res[2], False)
    print_result("DSATUR", heur_res[3], False)
    # 787s total time
    # 114s total time

    # 6.3
    for alg_name in mcts_res:
        print_result(alg_name, mcts_res[alg_name])

    print("Benchmark total time: ", total_time)
    
if __name__ == "__main__":
    main()
