import numpy as np
import numpy.typing as npt
import math


class DFA:
    """Representation of a DFA.
    :param initial_state: the initial state for the automa evaluation.
    :param transition_matrix: a 2D numpy array representation of the transition function (states on the rows and
        symbols on the cols).
    :param alphabet: the alphabet of symbols represented as a list of bytes, the byte is the encoding, while th index
        is the symbol in the transition matrix.
    :param accept: the set of the acceptance states.
    """

    def __init__(self, initial_state: int, transition_matrix: npt.NDArray, alphabet: bytes, accept: set):
        self.initial_state = initial_state
        self.transition_matrix = transition_matrix
        n_states, _ = transition_matrix.shape
        self.state_encoding_len = math.ceil(math.log2(n_states) / 8)  # The number of bytes needed to encode states.
        self.alphabet = alphabet
        self.accept = accept

    def encode_state(self, state: int) -> bytes:
        """Returns the encoding of the state using 'state_encoding_len' bytes."""
        return state.to_bytes(self.state_encoding_len, 'big')

    def encode_symbol(self, symbol: int) -> bytes:
        """Returns the encoding of the symbol."""
        return self.alphabet[symbol].to_bytes(1, 'big')

    def output(self, state: int, symbol: int) -> bytes:
        """Returns the 0 if the next state from 'state' and 'symbol' is accepted."""
        next_state = self.transition_matrix[state, symbol]
        out = 1 if next_state in self.accept else 0
        return self.encode_state(out)
