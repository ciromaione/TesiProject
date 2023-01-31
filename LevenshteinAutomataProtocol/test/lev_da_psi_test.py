import json
import numpy as np

import src.utils as utils
import src.lev_da_psi as psi

alpha = utils.Alphabet("abcde")
expected_res = {('cd', 'dcb'), ('abc', 'bce'), ('cd', 'bce'), ('cd', 'cda'), ('cd', 'bcda')}


def test_alice():
    with open("test/test_data/alice.txt", "r") as a1, open("test/test_data/alice_dfa.json", "r") as a:
        astrings = [line.strip() for line in a1.readlines()]
        aj = json.load(a)
        alice = []
        for word, dfa in zip(astrings, aj):
            d = json.loads(dfa)
            alice.append((word, utils.DFA(
                d['initial_state'],
                np.array(d['trans_matrix']),
                alpha,
                set(d['acceptance_state'])
            )))
    server = psi.Alice(alice, alpha)
    server.start()
    assert set(server.result) == expected_res


def test_bob():
    with open("test/test_data/bob.txt") as b:
        bob = [line.strip() for line in b.readlines()]
    client = psi.Bob(bob, "localhost")
    client.start()
    assert set(client.result) == expected_res
