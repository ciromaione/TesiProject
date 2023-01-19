from src.communication import ServerSocket, ClientSocket
from phe import paillier

m1 = {
    "test": "prova",
    "a": 1,
    "list": [1, 2, 3]
}

m2 = {
    "prova": 10,
    "arr": b'fasddklsdfjk'
}

pk, sk = paillier.generate_paillier_keypair()


def test_server():
    socket = ServerSocket(9000)
    data = socket.recv()
    assert data == m1
    n = socket.send_wait(m2)
    pk2 = paillier.PaillierPublicKey(n)
    socket.send(pk2.encrypt(10))


def test_client():
    socket = ClientSocket('localhost', 9000)
    data = socket.send_wait(m1)
    assert data == m2
    c = socket.send_wait(pk.n)
    assert sk.decrypt(c) == 10
