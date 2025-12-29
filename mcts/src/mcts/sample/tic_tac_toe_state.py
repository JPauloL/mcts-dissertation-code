from collections.abc import Iterator
from math import prod
from typing import cast

from mcts.interfaces import State


class TicTacToeStateIterator:
    state: TicTacToeState
    _iter_index: int # Index for iterator

    def __init__(self, state: TicTacToeState) -> None:
        self.state = state
        self._iter_index = 0

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self) -> int:
        if not self.state.is_terminal():
            while self._iter_index < 9:
                
                if self.state.board[self._iter_index] != 0:
                    self._iter_index += 1
                    continue
                
                action = self._iter_index
                self._iter_index += 1
                return action

        raise StopIteration

# type TttState = TicTacToeState # Alias needed to refer to the class when subclassing State

class TicTacToeState(State[int, float, float]):
    board: list[int]
    turn: int

    _hash: int | None = None

    def __init__(self, board: list[int] = [0] * 9, turn: int = 1) -> None:
        self.board = board
        self.turn = turn

    def actions_tree(self) -> Iterator[int]:
        return TicTacToeStateIterator(self)
    
    def actions_default(self) -> Iterator[int]:
        return TicTacToeStateIterator(self)

    def play(self, action: int) -> TicTacToeState:
        if action < 0 or action > 8 or self.board[action] != 0:
            return self
        
        new_board = self.board.copy()
        new_board[action] = self.turn
        return TicTacToeState(new_board, (self.turn % 2) + 1)

    @staticmethod
    def check_sequence(s1, s2, s3):
        return s1 != 0 and s1 == s2 == s3

    def winner(self):
        for i in range(3):
            if self.check_sequence(self.board[i * 3], self.board[i * 3 + 1], self.board[i * 3 + 2]):
                return self.board[i * 3]
            
            if self.check_sequence(self.board[i], self.board[3 + i], self.board[6 + i]):
                return self.board[i]

        if self.check_sequence(self.board[0], self.board[4], self.board[8]):
            return self.board[0]

        if self.check_sequence(self.board[2], self.board[4], self.board[6]):
            return self.board[2]
        
        # If there's no winner and there are empty spaces (prod = 0), the state is not terminal
        return -1 if prod(self.board) == 0 else 0 

    def is_terminal(self) -> bool:
        return self.winner() != -1

    def interpret_reward(self, reward: float) -> float:
        return -1 * reward if self.turn == 1 else reward

    def get_reward(self) -> float:
        winner = self.winner()

        if winner == 1:
            return 1
        elif winner == 2:
            return -1
        
        return 0
    
    @staticmethod
    def get_symbol(num: int) -> str:
        return "X" if num == 1 else ("O" if num == 2 else " ")
    
    def get_symbol_board(self) -> list[str]:
        return list(map(lambda x: self.get_symbol(x), self.board))

    def __str__(self) -> str:
        symbol_board = self.get_symbol_board()

        first_line = f" {symbol_board[0]} | {symbol_board[1]} | {symbol_board[2]}"
        second_line = f" {symbol_board[3]} | {symbol_board[4]} | {symbol_board[5]}"
        third_line = f" {symbol_board[6]} | {symbol_board[7]} | {symbol_board[8]}"

        sep_line = "---+---+---"

        return f"{first_line}\n{sep_line}\n{second_line}\n{sep_line}\n{third_line}\n"
    
    def equals(self, other: TicTacToeState) -> bool:
        return self.board == other.board

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        else:
            return hash(self.board) 
    
    def __eq__(self, other: object) -> bool:
        try:
            other_state = cast(TicTacToeState, other)
            return self.equals(other_state)
        except:
            return False
