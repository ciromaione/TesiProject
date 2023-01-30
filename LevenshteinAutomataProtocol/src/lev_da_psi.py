from phe import paillier

import src.utils as utils
import src.communication as com
import src.odfae as odfae

DEFAULT_PORT = 9000


class Alice:
    def __init__(
            self,
            dfa_list: list[(str, utils.DFA)],
            alphabet: utils.Alphabet,
            port: int = DEFAULT_PORT
    ):
        self.dfa_list = dfa_list
        self.alphabet = alphabet
        self.socket = com.ServerSocket(port)
        self.result = []

    def start(self):
        client_setup = self.socket.recv()
        print("client setup", client_setup)
        pk = client_setup["pk"]
        words_lens = client_setup["wlen"]
        server_setup = {
            "dfa_enc_len": [dfa.state_encoding_len for _, dfa in self.dfa_list],
            "alpha": self.alphabet
        }
        self.socket.send(server_setup)
        for word, dfa in self.dfa_list:
            dfa_garbler = odfae.Garbler(dfa, pk,  self.socket)
            for wlen in words_lens:
                dfa_garbler.serve_evaluation(wlen)
                res = self.socket.recv()
                if res:
                    self.socket.send(word)
                    self.result.append((word, res))
                else:
                    self.socket.send(False)


class Bob:
    def __init__(self, words_list: list[str], server: str, port: int = DEFAULT_PORT):
        self.words_list = words_list
        self.socket = com.ClientSocket(server, port)
        self.result = []

    def start(self):
        words_len = [len(w) for w in self.words_list]
        pk, sk = paillier.generate_paillier_keypair()
        client_setup = {
            "pk": pk,
            "wlen": words_len
        }
        server_setup = self.socket.send_wait(client_setup)
        dfa_enc_lens = server_setup["dfa_enc_len"]
        alphabet = server_setup["alpha"]
        for enc_len in dfa_enc_lens:
            evaluator = odfae.Evaluator(pk, sk, self.socket, alphabet, enc_len)
            for word in self.words_list:
                accepted = evaluator.evaluate(word)
                if accepted:
                    res = self.socket.send_wait(word)
                    self.result.append((word, res))
                else:
                    self.socket.send_wait(False)
