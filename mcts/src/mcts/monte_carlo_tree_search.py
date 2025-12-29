from mcts.interfaces import TreePolicy, DefaultPolicy

class MonteCarloTreeSearch[T, R]:

    def __init__(self, tree_policy: TreePolicy[T, R], default_policy: DefaultPolicy[T, R]):
        self.tree_policy = tree_policy
        self.default_policy = default_policy

    def run(self, root: T, max_iter: int, max_time: float=1000) -> T:
        n_iter = 0

        while n_iter < max_iter:
            node = self.tree_policy.tree_policy(root)

            res = self.default_policy.simulate(node)
            self.tree_policy.backpropagate(node, res)

            n_iter += 1
            print(f"Finished iteration {n_iter}")

        return root

    def __str__(self) -> str:
        return "Monte Carlo Tree Search!"
