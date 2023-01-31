import json
import numpy as np

import src.utils as utils
import src.lev_da_psi as psi

alpha = utils.Alphabet("abcde")
expected_res = {('cd', 'dcb'), ('abc', 'bce'), ('cd', 'bce'), ('cd', 'cda'), ('cd', 'bcda')}


def test_alice():
    with open("test/test_data/alice_dfa.json", "r") as a:
        alice = []
        json_dfas = json.load(a)
        for dfa in json_dfas:
            word = dfa["original_string"]
            d = utils.DFA(
                dfa["initial_state"],
                np.array(dfa["trans_matrix"]),
                alpha,
                set(dfa["acceptance_state"])
            )
            alice.append((word, d))
    server = psi.Alice(alice, alpha)
    server.start()
    assert set(server.result) == expected_res


def test_bob():
    with open("test/test_data/bob.txt") as b:
        bob = [line.strip() for line in b.readlines()]
    client = psi.Bob(bob, alpha, "localhost")
    client.start()
    assert set(client.result) == expected_res
