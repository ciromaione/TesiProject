import numpy.typing as npt
import numpy as np
import math


def ro_hash(data: bytes) -> bytes:
    return data


def permute(state: int, r: int, num_states: int):
    return int((state + r) % num_states)


class DFA:
    """Representation of a DFA.

    Args:
        initial_state: the initial state for the automa evaluation.
        transition_matrix: a 2D numpy array representation of the transition function (states on the rows and symbols on
                           the cols)
        alphabet: the alphabet of symbols represented as a list of bytes, the byte is the encoding, while th index is
                  the symbol in the transition matrix
        accept: the list of the acceptance states
    """

    def __init__(self, initial_state: int, transition_matrix: npt.NDArray, alphabet: bytes, accept: list):
        self.initial_state = initial_state
        self.transition_matrix = transition_matrix
        n_states, _ = transition_matrix.shape
        self.state_encoding_len = math.ceil(math.log2(n_states) / 8)  # The number of bytes needed to encode states.
        self.alphabet = alphabet
        self.accept = accept

    def encode_state(self, state: int) -> bytes:
        """Returns the encoding of the state using 'state_encoding_len' bytes."""
        return state.to_bytes(self.state_encoding_len, 'big')

    def encode_symbol(self, symbol: int):
        """Returns the encoding of the symbol."""
        return self.alphabet[symbol]


class GarbledDFA:
    """Representation of the Garbled form of a DFA.

    Args:
        plain_dfa: the DFA to garble.
        r: a random int used for the current permutation of the transaction matrix rows.
        keys: a list of keys for the encryption of states.
        next_r: the next random int for the permutation.
        next_keys: the next list of keys for the encryption.
    """
    def __init__(self, plain_dfa: DFA, r: int, keys: list, next_r: int, next_keys: list):
        self.plain_dfa = plain_dfa
        self.r = r
        self.keys = keys
        self.next_r = next_r
        self.next_keys = next_keys
        self._perform_garbling()

    def _perform_garbling(self):
        """Calculate the garbled transaction matrix."""
        rows, cols = self.plain_dfa.transition_matrix.shape
        self.garbled_matrix = np.empty((rows, cols), dtype=bytes)  # The empty garbled matrix.
        permuted_states = [permute(i, self.r, rows) for i in range(rows)]  # Permuted states using the random r.
        for q, qp in enumerate(permuted_states):  # q is the real state, qp is the permuted state.
            for ch in range(cols):
                # encryption
                x1 = ro_hash(self.keys[qp] + self.plain_dfa.encode_symbol(ch))
                next_perm_q = permute(self.plain_dfa.transition_matrix[q][ch], self.next_r, rows)
                x2 = (self.plain_dfa.encode_state(next_perm_q) + self.next_keys[next_perm_q])
                self.garbled_matrix[qp][ch] = bytes(b1 ^ b2 for b1, b2 in zip(x1, x2))  # byte-by-byte xor.
