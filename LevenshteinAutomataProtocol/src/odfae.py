import numpy as np
import numpy.typing as npt
import concurrent.futures as cf
import os
import typing
from phe import paillier

import src.utils as utils
import src.ot as ot
import src.communication as com

KEY_LEN = 16


def ro_hash(data: bytes) -> bytes:
    return data


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


class GarbledDFA:
    def __init__(self, dfa: utils.DFA, keys: npt.NDArray, r: tuple, word_len: int):
        self.dfa = dfa
        self.keys = keys
        self.r = r
        self.word_len = word_len
        self._max_workers = os.cpu_count() + 4
        self._generate_garbled_arrays()

    def _generate_garbled_arrays(self):
        futures_set = set()
        with cf.ProcessPoolExecutor(self._max_workers) as executor:
            for i in range(self.word_len):
                future = executor.submit(self._garble_matrix, i)
                futures_set.add(future)
            res = [None for _ in range(self.word_len)]
            for future in cf.as_completed(futures_set):
                i, arr = future.result()
                res[i] = arr
            self.garbled_arrays = res

    def _garble_matrix(self, index):
        rows, cols = self.dfa.transition_matrix.shape
        gm = [[b'' for _ in range(cols)] for _ in range(rows)]
        for q in range(rows):
            for sigma in range(cols):
                q1 = permute(self.r[index], q, rows)
                x1 = ro_hash(self.keys[q1, index] + self.dfa.encode_symbol(sigma))
                if index != self.word_len - 1:
                    next_perm_q = permute(
                        self.dfa.transition_matrix[q, sigma],
                        self.r[index + 1],
                        rows
                    )
                    x2 = self.dfa.encode_state(next_perm_q) + self.keys[next_perm_q, index + 1]
                else:
                    x2 = np.random.bytes(KEY_LEN) + self.dfa.output(q, sigma)
                gm[q1][sigma] = xor(x1, x2)
        return index, np.array(gm)


class Garbler:
    def __init__(self, dfa: utils.DFA, pk: paillier.PaillierPublicKey):
        self.dfa = dfa
        self._max_workers = os.cpu_count() + 4
        self.pk = pk

    def start_oe(self, word_len, ot_ports: list[int]) -> bytes:
        n_states, cols = self.dfa.transition_matrix.shape
        keys = np.array([[np.random.bytes(KEY_LEN) for _ in range(word_len)] for _ in range(n_states)])
        r = tuple(np.random.randint(0, word_len) for _ in range(word_len))
        gdfa = GarbledDFA(self.dfa, keys, r, word_len)

        futures_set = set()
        with cf.ProcessPoolExecutor(self._max_workers) as executor:
            for port, gmatrix in zip(ot_ports, gdfa.garbled_arrays):
                future = executor.submit(self._perform_ot, port, cols, gmatrix)
                futures_set.add(future)

            cf.wait(futures_set)

        return keys[permute(r[0], 0, n_states, True), 0]

    def _perform_ot(self, port, n, gmatrix):
        socket = com.ServerSocket(port)
        sender = ot.OTSender(n, socket, self.pk)
        sender.send_secrets(gmatrix)


class Evaluator:
    def __init__(self, pk: paillier.PaillierPublicKey, sk: paillier.PaillierPrivateKey, server_ip: str):
        self.pk = pk
        self.sk = sk
        self.server_ip = server_ip
        self._max_workers = os.cpu_count() + 4

    def recv_enc_matrix(self, word: str, ot_ports: list[int]) -> npt.NDArray:
        futures_set = set()
        len_word = len(word)
        with cf.ProcessPoolExecutor(self._max_workers) as executor:
            for i in range(len_word):
                future = executor.submit(self._perform_ot, i, ot_ports[i], len_word, word[i])
                futures_set.add(future)

            res = [None for _ in range(len(word))]
            for future in cf.as_completed(futures_set):
                i, col = future.result()
                res[i] = col

            return np.array(res).transpose()

    def _perform_ot(self, i, port, n, x, len_enc):
        socket = com.ClientSocket(self.server_ip, port)
        receiver = ot.OTReceiver(n, socket, self.pk, self.sk, len_enc)
        col = receiver.recv_secret(x)
        return i, col

    @staticmethod
    def evaluate(word: str, k0: bytes, enc_matrix: npt.NDArray) -> bool:
        n_states, n_cols = enc_matrix.shape
        len_enc = len(enc_matrix[0, 0]) - KEY_LEN

        next_keys = []
        x1 = ro_hash(k0 + ord(word[0]).to_bytes(1, 'big'))
        for g in enc_matrix[:, 0]:
            deg = xor(x1, g)
            q, nk = split_key(len_enc, deg)
            if q < n_states:
                next_keys.append((q, nk))

        for i in range(1, n_cols):
            tmp_keys = []
            for q, key in next_keys:
                x1 = ro_hash(key + ord(word[i]).to_bytes(1, 'big'))
                deg = xor(x1, enc_matrix[q, i])
                nq, nk = split_key(len_enc, deg)
                if q < n_states:
                    tmp_keys.append((nq, nk))
            next_keys = tmp_keys

        for res, _ in next_keys:
            if res == 1:
                return True
            elif res == 0:
                return False

        raise EvaluationException


class EvaluationException(Exception):
    """Raises when the evaluation doesn't produce a valid result."""
    pass
