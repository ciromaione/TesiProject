import numpy as np
import numpy.typing as npt
import concurrent.futures as cf
from phe import paillier
import hashlib

import src.utils as utils
import src.ot as ot
import src.communication as com

KEY_LEN = 16


def gen_random_key() -> bytes:
    """Generate a random key of KEY_LEN bytes."""
    while True:
        k = np.random.bytes(KEY_LEN)
        if len(k) == KEY_LEN:
            return k


def ro_hash(data: bytes, len_enc: int) -> bytes:
    """
    Performs the hash.
    :param data: input data.
    :param len_enc: length of the encoding of a state.
    :return: a digest of length KEY_LEN + len_enc, using the 'shake' hash function of the SHA-3 family.
    """
    return hashlib.shake_256(data).digest(KEY_LEN + len_enc)


def xor(x1: bytes, x2: bytes) -> bytes:
    """Performs the byte-to-byte xor of the given inputs.
    :param x1: the first operand.
    :param x2: the second operand.
    :return: x1 ^ x2.
    :raise Exception: if the length of x1 and x2 is different.
    """
    if len(x1) != len(x2):
        raise Exception(f"Inputs of the xor must have the same length.\nx1={len(x1)} --- {x1}\nx2={len(x2)} --- {x2}")
    return bytes(b1 ^ b2 for b1, b2 in zip(x1, x2))


def permute(r: int, st: int, n_st: int) -> int:
    """
    Permutation for the garbling/degarbling.
    :param r: the r value of the permutation.
    :param st: the state to permute.
    :param n_st: the number of states.
    :return: the permuted state.
    """
    return int((st + r) % n_st)


def split_key(len_enc: int, data: bytes) -> (int, bytes):
    """
    Splits decrypted data into state, key.
    :param len_enc: length of the encoding of a state.
    :param data: data to split.
    :return: the int state and the byte-array key.
    """
    q = int.from_bytes(data[:len_enc], 'big')
    nk = data[-KEY_LEN:]
    return q, nk


class Garbler:
    """
    Represents the Garbler party in the odfae protocol.
    :param dfa: a finite state automata.
    :param pk: the public key for the ot protocol.
    :param socket: a server socket for communication.
    """
    def __init__(self, dfa: utils.DFA, pk: paillier.PaillierPublicKey, socket: com.ServerSocket):
        self.dfa = dfa
        self.pk = pk
        self.socket = socket

    def serve_evaluation(self, word_len: int):
        """
        Generate the garbled arrays for the evaluation of a word.
        :param word_len: the length of the word.
        """
        n_states, cols = self.dfa.transition_matrix.shape
        # Generation of the |Q|·n random keys.
        keys = np.array([[gen_random_key() for _ in range(word_len)] for _ in range(n_states)])
        # Generation of the random int array for the permutations.
        r = tuple(np.random.randint(0, word_len) for _ in range(word_len))
        # Generation of the n garbled arrays.
        garbled_arrays = self._generate_garbled_arrays(keys, r, word_len)
        # Sending the right columns to the evaluator with OT.
        sender = ot.OTSender(self.dfa.alphabet.length, self.socket, self.pk)
        sender.send_secrets(garbled_arrays)
        # Sending the first key to the evaluator.
        k0 = keys[permute(r[0], 0, n_states), 0]
        self.socket.send(k0)

    def _generate_garbled_arrays(self, keys: npt.NDArray, r: tuple, word_len: int) -> list[npt.NDArray]:
        """
        Generates the garbled arrays.
        :param keys: the matrix of keys.
        :param r: the array of random integers for permutation.
        :param word_len: the length of the word.
        :return: the list of garbled arrays.
        """
        # Setup of the process pool executor for multiprocessing.
        futures_set = set()
        with cf.ProcessPoolExecutor(utils.MAX_WORKERS) as executor:
            for i in range(word_len):
                # Submits the construction of a single matrix as a parallel task.
                future = executor.submit(self._garble_matrix, i, self.dfa, keys, r, word_len)
                futures_set.add(future)
            res = [None] * word_len
            # Waiting for the results from the completing of the tasks.
            for future in cf.as_completed(futures_set):
                i, arr = future.result()
                res[i] = arr
            return res

    @staticmethod
    def _garble_matrix(index: int, dfa: utils.DFA, keys: npt.NDArray, r: tuple, word_len: int):
        """
        The worker for the i-th garbled matrix.
        :param index: matrix index.
        :param dfa: the starting dfa.
        :param keys: keys for the garbling.
        :param r: random integers for permutation.
        :param word_len: length of the word.
        :return: the i-th garbled matrix for the dfa.
        """
        rows, cols = dfa.transition_matrix.shape
        gm = [[b'' for _ in range(cols)] for _ in range(rows)]  # An empty 2-d array for the garbled matrix.
        for q in range(rows):
            for sigma in range(cols):
                q1 = permute(r[index], q, rows)  # The permuted state q.
                # First component for the encryption.
                x1 = ro_hash(keys[q1, index] + dfa.alphabet.encode(sigma), dfa.state_encoding_len)
                if index != word_len - 1:  # Checking if we are garbling the last column.
                    # The next state permuted with the next permutation.
                    next_perm_q = permute(
                        dfa.transition_matrix[q, sigma],
                        r[index + 1],
                        rows
                    )
                    # Second component of the encryption.
                    x2 = dfa.encode_state(next_perm_q) + keys[next_perm_q, index + 1]
                else:
                    x2 = b'\x00' * KEY_LEN + dfa.output(q, sigma)  # In the last column we garble the outputs.
                gm[q1][sigma] = xor(x1, x2)
        return index, np.array(gm)


class Evaluator:
    """
    Represents the evaluator party of the odfae protocol.
    :param pk: public key for the OT.
    :param sk: secret key for the OT.
    :param socket: client socket for communication.
    :param alphabet: the alphabet for words.
    :param len_enc_state: encoding length of the states.
    """
    def __init__(
            self,
            pk: paillier.PaillierPublicKey,
            sk: paillier.PaillierPrivateKey,
            socket: com.ClientSocket,
            alphabet: utils.Alphabet,
            len_enc_state: int
    ):
        self.pk = pk
        self.sk = sk
        self.socket = socket
        self.alphabet = alphabet
        self.len_enc_cell = len_enc_state + KEY_LEN

    def evaluate(self, word: str) -> bool:
        """
        Performs the evaluation of the word with the server.
        :param word: a string to evaluate.
        :return: true if the word is accepted, false otherwise.
        """
        choices = [self.alphabet.decode(c) for c in word]  # The list in which each character is a choice for ot.
        # The OT receiver.
        receiver = ot.OTReceiver(self.alphabet.length, self.socket, self.pk, self.sk, self.len_enc_cell)
        enc_matrix = receiver.recv_secrets(choices)  # The encrypted columns received with OT.
        k0 = self.socket.recv()  # The starting key for degarbling.
        return self._perform_degarbling(word, k0, enc_matrix)

    @staticmethod
    def _perform_degarbling(word: str, k0: bytes, enc_matrix: npt.NDArray) -> bool:
        """
        Performs the degarbling.
        :param word: the string to evaluate.
        :param k0: the starting key.
        :param enc_matrix: the encrypted matrix.
        :return: true if the word is accepted, false otherwise.
        """
        n_states, n_cols = enc_matrix.shape
        len_enc = len(enc_matrix[0, 0]) - KEY_LEN  # The length of the encoding of a state.

        next_keys = []
        # Performs the degarbling of the first column, that could generate different valid next keys and state,
        # so we proceed with all of those.
        x1 = ro_hash(k0 + word[0].encode(), len_enc)
        for g in enc_matrix[:, 0]:
            deg = xor(x1, g)
            nq, nk = split_key(len_enc, deg)
            if nq < n_states:
                next_keys.append((nq, nk))

        # Degarbling the next columns, except the last one.
        for i in range(1, n_cols - 1):
            tmp_keys = []
            for q, key in next_keys:
                x1 = ro_hash(key + word[i].encode(), len_enc)
                deg = xor(x1, enc_matrix[q, i])
                nq, nk = split_key(len_enc, deg)
                if nq < n_states:
                    tmp_keys.append((nq, nk))
            next_keys = tmp_keys
        # Degarbling the last column and extract the output.
        # After all the previous steps the probability of having more than one valid outputs is negligible.
        for q, key in next_keys:
            x1 = ro_hash(key + word[n_cols - 1].encode(), len_enc)
            deg = xor(x1, enc_matrix[q, n_cols - 1])
            out = int.from_bytes(deg, 'big')
            if out == 0:
                return False
            elif out == 1:
                return True
        raise EvaluationException  # Some unexpected error occurred.


class EvaluationException(Exception):
    """Raises when the evaluation doesn't produce a valid result."""
    pass
