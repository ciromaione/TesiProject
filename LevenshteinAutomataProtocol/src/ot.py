import numpy as np
import numpy.typing as npt
from phe import paillier
import concurrent.futures as cf

import src.communication as com
from src.utils import MAX_WORKERS
from src.utils import SymEnc

# The columns to send with OT will be split into chunks of 256 bytes (2048 bits)
CHUNKS_LEN = 256


def encode_message(array: npt.NDArray) -> bytes:
    """
    Performs the encoding of a matrix column into a list of integers for the encryption.
    :param array: the column to encode.
    :return: the encoded array in the form of a dict '{"lc": _, "vals": []}' where lc is the length in byte of the last
        chunk and vals is the list of the integer form of the bytes chunks.
    """
    data = b''
    for el in array:
        data += el
    return data


def decode_message(data: bytes, len_enc: int) -> npt.NDArray:
    """
    Performs the decoding of an encoded matrix column.
    :param data: the encoded dict in the form of '{"lc": _, "vals": []}' where lc is the length in byte of the last
        chunk and vals is the list of the integer form of the bytes chunks.
    :param len_enc: length of the encoding for the garbled cells of the matrix.
    :return: the original column matrix.
    """
    raw = data
    mess = [raw[i:i + len_enc] for i in range(0, len(raw), len_enc)]
    return np.array(mess, dtype=object)


class OTSender:
    """
    Representation of the Sender party in the one-out-of-N OT src
    :param n: the number of secrets to exchange.
    :param socket: the ServerSocket for the tcp communication.
    :param pk: a public key of a Paillier encryption scheme.
    """
    def __init__(self, n: int, socket: com.ServerSocket, pk: paillier.PaillierPublicKey):
        self.n = n
        self.socket = socket
        self.public_key = pk

    def send_secrets(self, garbled_arrays: list[npt.NDArray]):
        """
        Sends to the client the columns he has chosen.
        :param garbled_arrays: a list of np 2D arrays in which the columns will be the secrets.
        """
        choice_bits_list = self.socket.recv()  # the list of encrypted choice bit-vectors received from the client
        instances = len(choice_bits_list)  # number of instances of ot needed to exchange secrets.

        # Setup of the process pool executor for multiprocessing.
        futures_set = set()
        with cf.ProcessPoolExecutor(MAX_WORKERS) as executor:
            for i in range(instances):
                future = executor.submit(  # Submitting encryption of a secret as a parallel task.
                    self._encrypt_secret,
                    self.n,
                    self.public_key,
                    i,
                    garbled_arrays[i],
                    choice_bits_list[i]
                )
                futures_set.add(future)

        # Waiting for the results from the completing of the tasks.
        chosen_keys = [None] * instances
        enc_matrices = [None] * instances
        for future in cf.as_completed(futures_set):
            i, chosen_key, enc_cols = future.result()
            chosen_keys[i] = chosen_key
            enc_matrices[i] = enc_cols

        self.socket.send_wait((chosen_keys, enc_matrices))  # sending the encrypted columns.

    @staticmethod
    def _encrypt_secret(
            n: int,
            pk: paillier.PaillierPublicKey,
            index: int,
            matrix: npt.NDArray,
            choice_bits: list[paillier.EncryptedNumber]
    ):
        """
        Performs the encryption of the secret to share with client.
        :param n: number of secrets.
        :param pk: public key of an AHE scheme.
        :param index: index.
        :param matrix: matrix in witch the columns are the secrets.
        :param choice_bits: encrypted choice bits received form the client.
        :return: the i-th chosen encrypted secret.
        """
        symenc = SymEnc()
        secrets = tuple(symenc.genKey() for _ in range(n))
        encrypted_cols = tuple(symenc.encrypt(secrets[i], encode_message(matrix[:, i])) for i in range(n))

        chosen_key = None
        for b, s in zip(choice_bits, secrets):
            s = int.from_bytes(s, 'big')
            if chosen_key:
                chosen_key += b * s
            else:
                chosen_key = b * s

        chosen_key += pk.encrypt(0)

        return index, chosen_key, encrypted_cols


class OTReceiver:
    """
    Representation of the Receiver party in the one-out-of-N OT src
    :param n: the number of secrets to exchange.
    :param socket: the ClientSocket for the tcp communication.
    :param pk: a public key of a Paillier encryption scheme.
    :param sk: a secret key of a Paillier encryption scheme.
    :param len_encoding_states: length of the encoding for the garbled cells of the matrix.
    """
    def __init__(
            self,
            n: int,
            socket: com.ClientSocket,
            pk: paillier.PaillierPublicKey,
            sk: paillier.PaillierPrivateKey,
            len_encoding_states: int
    ):
        self.public_key = pk
        self.secret_key = sk
        self.n = n
        self.socket = socket
        self.len_enc_states = len_encoding_states

    def recv_secrets(self, choices: list[int]) -> npt.NDArray:
        """
        Request the chosen secret.
        :param choices: list of secret numbers requested, 0 <= choice < n.
        :return: the chosen columns.
        """
        # Setup of the process pool executor for multiprocessing.
        futures_set = set()
        with cf.ProcessPoolExecutor(MAX_WORKERS) as executor:
            for i, choice in enumerate(choices):
                # Submitting the encryption of the choice as a parallel task.
                future = executor.submit(self._encrypt_choice, self.public_key, self.n, i, choice)
                futures_set.add(future)

            # Waiting for the results from the completing of the tasks.
            enc_choices = [None] * len(choices)
            for future in cf.as_completed(futures_set):
                i, enc_choice = future.result()
                enc_choices[i] = enc_choice

            chosen_keys, enc_matrices = self.socket.send_wait(enc_choices)  # The ciphertexts received from server.
            self.socket.send(True)  # For socket sync.
            futures_set = set()
            for i, choice in enumerate(choices):
                # Submitting the decryption of a column as a parallel task.
                future = executor.submit(self._decrypt_col, self.secret_key, self.len_enc_states, i, choice, chosen_keys[i], enc_matrices[i])
                futures_set.add(future)

            # Waiting for the results from the completing of the tasks.
            res = [None] * len(choices)
            for future in cf.as_completed(futures_set):
                i, enc_col = future.result()
                res[i] = enc_col

            return np.array(res, dtype=object).transpose()

    @staticmethod
    def _encrypt_choice(pk: paillier.PaillierPublicKey, n: int, index: int, choice: int):
        """
        Worker for the encryption of a choice.
        :param pk: public key.
        :param n: n.
        :param index: index.
        :param choice: the choice.
        :return: the encrypted choice as a list of ciphertexts.
        """
        # encode the choice as a vector of n values with 1 in position choice and 0 otherwise.
        enc_choice = [pk.encrypt(1 if i == choice else 0) for i in range(n)]
        return index, enc_choice

    @staticmethod
    def _decrypt_col(sk: paillier.PaillierPrivateKey, len_enc_states: int, index: int, choice, chosen_key, enc_matrix):
        """
        Worker for decryption of a column.
        :param sk: secret key.
        :param len_enc_states: encoding length of the states.
        :param index: index.
        :param ciphertext: the received ciphertext.
        :return: the decrypted column.
        """

        symenc = SymEnc()
        key = sk.decrypt(chosen_key).to_bytes(symenc.KEY_SIZE, 'big')
        col = decode_message(symenc.decrypt(key, enc_matrix[choice]), len_enc_states)
        return index, col
