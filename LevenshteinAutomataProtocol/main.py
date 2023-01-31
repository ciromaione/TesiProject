import sys
import numpy as np
import json

import src.utils as utils
import src.lev_da_psi as psi

SERVER_HOST = "localhost"
SERVER_PORT = 9000


def alice(file_name, alpha):
    dfas = []
    with open(file_name, 'r') as json_file:
        json_dfas = json.load(json_file)
        for dfa in json_dfas:
            word = dfa["original_string"]
            d = utils.DFA(
                dfa["initial_state"],
                np.array(dfa["trans_matrix"]),
                alpha,
                set(dfa["acceptance_state"])
            )
            dfas.append((word, d))
    server = psi.Alice(dfas, alpha, SERVER_PORT)
    server.start()
    print("Res Server:", server.result)


def bob(file_name, alpha):
    with open(file_name, 'r') as f:
        words = [line.strip() for line in f.readlines()]
    client = psi.Bob(words, alpha, SERVER_HOST, SERVER_PORT)
    client.start()
    print("Res Client:", client.result)


def print_usage():
    print("Usage: [ python3 main.py role file_name alphabet ]")
    print("\t - role: 'alice' or 'bob'.")
    print("\t - file_name: the json file of DFAs for alice, the strings for bob.")
    print("\t - alphabet: a string with all the characters in the alphabet.")
    sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print_usage()
    _, role, file, alpha_str = sys.argv
    alpha = utils.Alphabet(alpha_str)
    if role == "alice":
        alice(file, alpha)
    elif role == "bob":
        bob(file, alpha)
    else:
        print_usage()


if __name__ == '__main__':
    main()
