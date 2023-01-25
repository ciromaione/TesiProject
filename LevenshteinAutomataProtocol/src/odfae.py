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
    while True:
        k = np.random.bytes(KEY_LEN)
        if len(k) == KEY_LEN:
            return k


def ro_hash(data: bytes, len_enc: int) -> bytes:
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


def permute(r: int, st: int, n_st: int, inv=False) -> int:
    if inv:
        return int((st - r) % n_st)
    return int((st + r) % n_st)


def split_key(len_enc: int, data: bytes) -> (int, bytes):
    q = int.from_bytes(data[:len_enc], 'big')
    nk = data[-KEY_LEN:]
    return q, nk


class Garbler:
    def __init__(self, dfa: utils.DFA, pk: paillier.PaillierPublicKey, socket: com.ServerSocket):
        self.dfa = dfa
        self.pk = pk
        self.socket = socket

    def serve_evaluation(self, word_len: int):
        n_states, cols = self.dfa.transition_matrix.shape
        keys = np.array([[gen_random_key() for _ in range(word_len)] for _ in range(n_states)])
        r = tuple(np.random.randint(0, word_len) for _ in range(word_len))
        garbled_arrays = self._generate_garbled_arrays(keys, r, word_len)
        sender = ot.OTSender(self.dfa.alphabet.length, self.socket, self.pk)
        sender.send_secrets(garbled_arrays)
        k0 = keys[permute(r[0], 0, n_states), 0]
        self.socket.send(k0)

    def _generate_garbled_arrays(self, keys: npt.NDArray, r: tuple, word_len: int) -> list[npt.NDArray]:
        futures_set = set()
        with cf.ProcessPoolExecutor(utils.MAX_WORKERS) as executor:
            for i in range(word_len):
                future = executor.submit(self._garble_matrix, i, self.dfa, keys, r, word_len)
                futures_set.add(future)
            res = [None] * word_len
            for future in cf.as_completed(futures_set):
                i, arr = future.result()
                res[i] = arr
            return res

    @staticmethod
    def _garble_matrix(index: int, dfa: utils.DFA, keys: npt.NDArray, r: tuple, word_len: int):
        rows, cols = dfa.transition_matrix.shape
        gm = [[b'' for _ in range(cols)] for _ in range(rows)]
        for q in range(rows):
            for sigma in range(cols):
                q1 = permute(r[index], q, rows)
                x1 = ro_hash(keys[q1, index] + dfa.alphabet.encode(sigma), dfa.state_encoding_len)
                if index != word_len - 1:
                    next_perm_q = permute(
                        dfa.transition_matrix[q, sigma],
                        r[index + 1],
                        rows
                    )
                    x2 = dfa.encode_state(next_perm_q) + keys[next_perm_q, index + 1]
                else:
                    x2 = b'\x00' * KEY_LEN + dfa.output(q, sigma)
                gm[q1][sigma] = xor(x1, x2)
        return index, np.array(gm)


class Evaluator:
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
        choices = [self.alphabet.decode(c) for c in word]
        receiver = ot.OTReceiver(self.alphabet.length, self.socket, self.pk, self.sk, self.len_enc_cell)
        enc_matrix = receiver.recv_secrets(choices)
        k0 = self.socket.recv()
        return self._perform_degarbling(word, k0, enc_matrix)

    @staticmethod
    def _perform_degarbling(word: str, k0: bytes, enc_matrix: npt.NDArray) -> bool:
        n_states, n_cols = enc_matrix.shape
        len_enc = len(enc_matrix[0, 0]) - KEY_LEN

        next_keys = []
        x1 = ro_hash(k0 + word[0].encode(), len_enc)
        for g in enc_matrix[:, 0]:
            deg = xor(x1, g)
            nq, nk = split_key(len_enc, deg)
            if nq < n_states:
                next_keys.append((nq, nk))

        for i in range(1, n_cols - 1):
            tmp_keys = []
            for q, key in next_keys:
                x1 = ro_hash(key + word[i].encode(), len_enc)
                deg = xor(x1, enc_matrix[q, i])
                nq, nk = split_key(len_enc, deg)
                if nq < n_states:
                    tmp_keys.append((nq, nk))
            next_keys = tmp_keys
        for q, key in next_keys:
            x1 = ro_hash(key + word[n_cols - 1].encode(), len_enc)
            deg = xor(x1, enc_matrix[q, n_cols - 1])
            out = int.from_bytes(deg, 'big')
            if out == 0:
                return False
            elif out == 1:
                return True
        raise EvaluationException


class EvaluationException(Exception):
    """Raises when the evaluation doesn't produce a valid result."""
    pass
