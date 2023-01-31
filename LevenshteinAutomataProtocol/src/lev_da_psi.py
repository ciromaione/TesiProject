from phe import paillier

import src.utils as utils
import src.communication as com
import src.odfae as odfae

DEFAULT_PORT = 9000


class Alice:
    """
    Representation of Alice party in the protocol.
    :param dfa_list: list of DFA representation of Alice strings.
    :param alphabet: the alphabet.
    :param port: the server port.
    """
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
        client_setup = self.socket.recv()  # receiving setup from Bob.
        pk = client_setup["pk"]  # Paillier public key.
        words_lens = client_setup["wlen"]  # length of Bob words.
        server_setup = {
            "dfa_enc_len": [dfa.state_encoding_len for _, dfa in self.dfa_list]
        }
        self.socket.send(server_setup)  # sending setup to Bob.
        dfa_count = 1
        for word, dfa in self.dfa_list:  # for all DFAs.
            dfa_garbler = odfae.Garbler(dfa, pk,  self.socket)
            for i, wlen in enumerate(words_lens):
                print(f"DFA {dfa_count}/{len(self.dfa_list)}, word {i}/{len(words_lens)}", end='\r')
                dfa_count += 1
                dfa_garbler.serve_evaluation(wlen)  # serve the evaluation for each Bob's word.
                res = self.socket.recv()
                if res:
                    self.socket.send(word)
                    self.result.append((word, res))
                else:
                    self.socket.send(False)


class Bob:
    """
    Representation of Bob party in the protocol.
    :param words_list: the word list input of psi.
    :param alpha: alphabet.
    :param server: server host.
    :param port: server port.
    """
    def __init__(
            self,
            words_list: list[str],
            alpha: utils.Alphabet,
            server: str,
            port: int = DEFAULT_PORT
    ):
        self.words_list = words_list
        self.socket = com.ClientSocket(server, port)
        self.result = []
        self.alphabet = alpha

    def start(self):
        words_len = [len(w) for w in self.words_list]  # the length of the words to send to Alice.
        pk, sk = paillier.generate_paillier_keypair()  # Paillier keys generation.
        client_setup = {  # setup data for Alice
            "pk": pk,
            "wlen": words_len
        }
        server_setup = self.socket.send_wait(client_setup)  # sending setup to Alice and receiving her setup.
        dfa_enc_lens = server_setup["dfa_enc_len"]  # length of the encrypted entries for all DFAs.
        for enc_len in dfa_enc_lens:
            # Evaluation all strings on each DFA.
            evaluator = odfae.Evaluator(pk, sk, self.socket, self.alphabet, enc_len)
            for word in self.words_list:
                accepted = evaluator.evaluate(word)
                if accepted:
                    res = self.socket.send_wait(word)
                    self.result.append((word, res))
                else:
                    self.socket.send_wait(False)
