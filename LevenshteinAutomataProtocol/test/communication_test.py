from LevenshteinAutomataProtocol.src.communication import ServerSocket, ClientSocket

m1 = {
    "test": "prova",
    "a": 1,
    "list": [1, 2, 3]
}

m2 = {
    "prova": 10,
    "arr": b'fasddklsdfjk'
}


def test_server():
    socket = ServerSocket(9000)
    data = socket.recv()
    assert data == m1
    socket.send(m2)


def test_client():
    socket = ClientSocket('localhost', 9000)
    data = socket.send_wait(m1)
    assert data == m2
