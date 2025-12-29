### Folder Structure

#### src

- algorithms: Contains the implementation of some mcts algorithms (e.g. UCT). An algorithm is defined by mainly by its tree policy and tree node representation, some default policies can be reused by different algorithms;
- interfaces: Contains the interfaces that must be implemented to run a MCTS;
- other: Contains the Solutions class;
- sample: Contains sample states - namely two representations of the Graph Coloring problem and a representation of a TicTacToe game;

### Running Tests

```
uv run pytest
```