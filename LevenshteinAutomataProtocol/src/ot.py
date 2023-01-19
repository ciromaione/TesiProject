import numpy as np
import numpy.typing as npt
from phe import paillier

import src.communication as com

# The columns to send with OT will be split into chunks of 256 bytes (2048 bits)
CHUNKS_LEN = 256


def encode_message(array: npt.NDArray) -> dict:
    """
    Performs the encoding of a matrix column into a list of integers for the encryption.
    :param array: the column to encode.
    :return: the encoded array in the form of a dict '{"lc": _, "vals": []}' where lc is the length in byte of the last
        chunk and vals is the list of the integer form of the bytes chunks.
    """
    data = b''
    for el in array:
        data += el
    last_chunk_len = len(data) % CHUNKS_LEN
    values = []
    for i in range(0, len(data), CHUNKS_LEN):
        values.append(int.from_bytes(data[i: i + CHUNKS_LEN], 'big'))
    return {
        "lc": last_chunk_len,
        "vals": values
    }


def decode_message(data: dict, len_enc: int) -> npt.NDArray:
    """
    Performs the decoding of an encoded matrix column.
    :param data: the encoded dict in the form of '{"lc": _, "vals": []}' where lc is the length in byte of the last
        chunk and vals is the list of the integer form of the bytes chunks.
    :param len_enc: length of the encoding for the garbled cells of the matrix.
    :return: the original column matrix.
    """
    values = data["vals"]
    last = values.pop()
    raw = b''
    for v in values:
        try:
            raw += v.to_bytes(CHUNKS_LEN, 'big')
        except OverflowError:
            print("value: ", v)
    raw += last.to_bytes(data["lc"], 'big')
    mess = [raw[i:i + len_enc] for i in range(0, len(raw), len_enc)]
    return np.array(mess)


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

    def send_secrets(self, matrix: npt.NDArray):
        """
        Start the src for the OT.
        :param matrix: a np 2D array in which the columns will be the secrets.
        """
        choice_bits = self.socket.recv()  # the encrypted choice bit-vector received from the client
        secrets = tuple(encode_message(matrix[:, i]) for i in range(self.n))  # list of secrets encoded

        # for the encoding the ciphertext is split in chunks
        n_ciphertexts = len(secrets[0]["vals"])  # number of chunks
        ciphertexts = [self.public_key.encrypt(0) for _ in range(n_ciphertexts)]  # chunks initialized at 0

        # encryption of the chunks
        for b, s in zip(choice_bits, secrets):
            for i in range(n_ciphertexts):
                ciphertexts[i] += b * s["vals"][i]

        # encryption of the last chunk size
        last_chunk_size = self.public_key.encrypt(0)
        for i, b in enumerate(choice_bits):
            last_chunk_size += b * secrets[i]["lc"]

        self.socket.send({
            "lc": last_chunk_size,
            "vals": ciphertexts
        })


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

    def recv_secret(self, choice) -> npt.NDArray:
        """
        Request the chosen secret.
        :param choice: secret number requested 0 <= choice < n.
        :return: the chosen column.
        """
        # encode the choice as a vector of n values with 1 in position choice and 0 otherwise.
        encoded_choice = [self.public_key.encrypt(1 if i == choice else 0) for i in range(self.n)]
        ciphertext = self.socket.send_wait(encoded_choice)  # the ciphertext received from server.
        lc = self.secret_key.decrypt(ciphertext["lc"])  # decryption of the last chunk size.
        values = [self.secret_key.decrypt(v) for v in ciphertext["vals"]]  # decryption of the chunks.
        return decode_message({  # decoding the decrypted secret.
            "lc": lc,
            "vals": values
        }, self.len_enc_states)
